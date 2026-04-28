//! SIMD-accelerated dot products for the search hot path.
//!
//! Three dtype paths, three SIMD targets:
//!
//! ```text
//!                AVX2 (x86_64)         NEON (aarch64)        Scalar
//!   f32 · f32    8 lanes (f32x8)       4 lanes (f32x4)       autovec
//!   f32 · f16    8 lanes (load+cvt)    4 lanes (load+cvt)    autovec
//!   f32 · i8     16 lanes (i8 -> i32)  16 lanes (i8 -> i32)  autovec
//! ```
//!
//! Accumulators are always f32. The query is f32 (L2-normalized), the
//! database is f32 / f16 / i8 (i8 with a per-vector scale). Final score
//! is the real cosine.
//!
//! Detection happens once at module load via `OnceLock`. The dispatch
//! function is a function pointer chosen at first call, so the per-query
//! cost is one indirect call, not a CPUID check per vector.

use std::sync::OnceLock;

use nest_format::Int8EmbeddingsView;

/// What backend is the runtime using right now? Useful for `nest stats`
/// / benchmarks so the user can see whether SIMD is active.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SimdBackend {
    Scalar,
    Avx2,
    Neon,
}

impl SimdBackend {
    pub fn name(self) -> &'static str {
        match self {
            Self::Scalar => "scalar",
            Self::Avx2 => "avx2",
            Self::Neon => "neon",
        }
    }
}

static BACKEND: OnceLock<SimdBackend> = OnceLock::new();

/// The SIMD backend selected at runtime. Cached after the first call.
///
/// Set `NEST_FORCE_SCALAR=1` to disable SIMD entirely — useful for
/// before/after SIMD benchmarks on the same binary.
pub fn detect_backend() -> SimdBackend {
    *BACKEND.get_or_init(|| {
        if std::env::var("NEST_FORCE_SCALAR")
            .map(|v| v != "0")
            .unwrap_or(false)
        {
            return SimdBackend::Scalar;
        }
        #[cfg(target_arch = "x86_64")]
        {
            if std::is_x86_feature_detected!("avx2") && std::is_x86_feature_detected!("fma") {
                return SimdBackend::Avx2;
            }
        }
        #[cfg(target_arch = "aarch64")]
        {
            if std::arch::is_aarch64_feature_detected!("neon") {
                return SimdBackend::Neon;
            }
        }
        SimdBackend::Scalar
    })
}

// ---------------------------------------------------------------------------
// f32 · f32 (raw embeddings, mmap as &[u8])
// ---------------------------------------------------------------------------

/// Dot product between an f32 query and an f32 row stored as little-endian
/// bytes (the way embeddings live in mmap).
#[inline]
pub fn dot_f32_bytes(q: &[f32], row_bytes: &[u8]) -> f32 {
    debug_assert_eq!(row_bytes.len(), q.len() * 4);
    match detect_backend() {
        #[cfg(target_arch = "x86_64")]
        SimdBackend::Avx2 => unsafe { dot_f32_avx2(q, row_bytes) },
        #[cfg(target_arch = "aarch64")]
        SimdBackend::Neon => unsafe { dot_f32_neon(q, row_bytes) },
        _ => dot_f32_scalar(q, row_bytes),
    }
}

#[inline]
pub fn dot_f32_scalar(q: &[f32], row_bytes: &[u8]) -> f32 {
    let mut acc = 0.0f32;
    for (i, qv) in q.iter().enumerate() {
        let off = i * 4;
        let v = f32::from_le_bytes([
            row_bytes[off],
            row_bytes[off + 1],
            row_bytes[off + 2],
            row_bytes[off + 3],
        ]);
        acc += *qv * v;
    }
    acc
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn dot_f32_avx2(q: &[f32], row_bytes: &[u8]) -> f32 {
    unsafe {
        use std::arch::x86_64::*;
        let dim = q.len();
        let mut acc = _mm256_setzero_ps();
        let row_ptr = row_bytes.as_ptr() as *const f32;
        let chunks = dim / 8;
        for i in 0..chunks {
            let qv = _mm256_loadu_ps(q.as_ptr().add(i * 8));
            let rv = _mm256_loadu_ps(row_ptr.add(i * 8));
            acc = _mm256_fmadd_ps(qv, rv, acc);
        }
        let mut tail = 0.0f32;
        for (i, &qv) in q.iter().enumerate().skip(chunks * 8) {
            let off = i * 4;
            let v = f32::from_le_bytes([
                row_bytes[off],
                row_bytes[off + 1],
                row_bytes[off + 2],
                row_bytes[off + 3],
            ]);
            tail += qv * v;
        }
        let mut buf = [0.0f32; 8];
        _mm256_storeu_ps(buf.as_mut_ptr(), acc);
        buf.iter().sum::<f32>() + tail
    }
}

#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
unsafe fn dot_f32_neon(q: &[f32], row_bytes: &[u8]) -> f32 {
    unsafe {
        use std::arch::aarch64::*;
        let dim = q.len();
        let row_ptr = row_bytes.as_ptr() as *const f32;
        let mut acc = vdupq_n_f32(0.0);
        let chunks = dim / 4;
        for i in 0..chunks {
            let qv = vld1q_f32(q.as_ptr().add(i * 4));
            let rv = vld1q_f32(row_ptr.add(i * 4));
            acc = vfmaq_f32(acc, qv, rv);
        }
        let mut tail = 0.0f32;
        for (i, &qv) in q.iter().enumerate().skip(chunks * 4) {
            let off = i * 4;
            let v = f32::from_le_bytes([
                row_bytes[off],
                row_bytes[off + 1],
                row_bytes[off + 2],
                row_bytes[off + 3],
            ]);
            tail += qv * v;
        }
        let lane_sum = vaddvq_f32(acc);
        lane_sum + tail
    }
}

// ---------------------------------------------------------------------------
// f32 · f16 (float16 embeddings)
// ---------------------------------------------------------------------------

/// Dot product between an f32 query and an f16 row stored as little-endian
/// bytes. Accumulates in f32. The query stays f32 (it is normalized once
/// per call, no need to drop precision there).
#[inline]
pub fn dot_f32_f16_bytes(q: &[f32], row_bytes: &[u8]) -> f32 {
    debug_assert_eq!(row_bytes.len(), q.len() * 2);
    match detect_backend() {
        #[cfg(target_arch = "aarch64")]
        SimdBackend::Neon => unsafe { dot_f32_f16_neon(q, row_bytes) },
        // AVX2 has no native f16->f32 unless F16C is present; our cutoff
        // is "AVX2 + FMA" which usually pulls F16C along. Using a portable
        // unpack here keeps the AVX2 path simple and avoids the F16C
        // detection branch.
        _ => dot_f32_f16_scalar(q, row_bytes),
    }
}

#[inline]
pub fn dot_f32_f16_scalar(q: &[f32], row_bytes: &[u8]) -> f32 {
    let mut acc = 0.0f32;
    for (i, qv) in q.iter().enumerate() {
        let off = i * 2;
        let h = half::f16::from_le_bytes([row_bytes[off], row_bytes[off + 1]]);
        acc += *qv * h.to_f32();
    }
    acc
}

#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
unsafe fn dot_f32_f16_neon(q: &[f32], row_bytes: &[u8]) -> f32 {
    unsafe {
        // NEON has fcvtl to widen f16 -> f32 in groups of 4. Pack 4 lanes per
        // step. half::f16 layout matches IEEE binary16, same as ARM f16.
        use std::arch::aarch64::*;
        let dim = q.len();
        let row_ptr = row_bytes.as_ptr() as *const u16;
        let mut acc = vdupq_n_f32(0.0);
        let chunks = dim / 4;
        for i in 0..chunks {
            let halfs = vld1_u16(row_ptr.add(i * 4));
            // Reinterpret as float16x4_t and widen.
            let f16x4: float16x4_t = std::mem::transmute(halfs);
            let widened: float32x4_t = vcvt_f32_f16(f16x4);
            let qv = vld1q_f32(q.as_ptr().add(i * 4));
            acc = vfmaq_f32(acc, qv, widened);
        }
        let mut tail = 0.0f32;
        for (i, &qv) in q.iter().enumerate().skip(chunks * 4) {
            let off = i * 2;
            let h = half::f16::from_le_bytes([row_bytes[off], row_bytes[off + 1]]);
            tail += qv * h.to_f32();
        }
        let lane_sum = vaddvq_f32(acc);
        lane_sum + tail
    }
}

// ---------------------------------------------------------------------------
// f32 · i8 (int8 quantized embeddings, with per-vector scale)
// ---------------------------------------------------------------------------

/// Dot product between an f32 query and a single i8 row, multiplied by
/// the row's f32 scale. `q` stays f32; the i8 row is widened to i32 in
/// the inner loop, multiplied by f32 lanes of `q`, accumulated in f32.
///
/// `f32_value ≈ i8_value * scale`, so:
///   `q · v = scale * sum_i(q_i * i8_i)`.
#[inline]
pub fn dot_f32_i8(q: &[f32], row: &[i8], scale: f32) -> f32 {
    debug_assert_eq!(row.len(), q.len());
    let acc = match detect_backend() {
        #[cfg(target_arch = "x86_64")]
        SimdBackend::Avx2 => unsafe { dot_f32_i8_avx2(q, row) },
        #[cfg(target_arch = "aarch64")]
        SimdBackend::Neon => unsafe { dot_f32_i8_neon(q, row) },
        _ => dot_f32_i8_scalar(q, row),
    };
    acc * scale
}

#[inline]
pub fn dot_f32_i8_scalar(q: &[f32], row: &[i8]) -> f32 {
    let mut acc = 0.0f32;
    for (qv, &iv) in q.iter().zip(row.iter()) {
        acc += *qv * (iv as f32);
    }
    acc
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn dot_f32_i8_avx2(q: &[f32], row: &[i8]) -> f32 {
    unsafe {
        use std::arch::x86_64::*;
        let dim = q.len();
        let mut acc = _mm256_setzero_ps();
        let chunks = dim / 8;
        for i in 0..chunks {
            let i8_ptr = row.as_ptr().add(i * 8) as *const i64;
            // Load 8 i8s (8 bytes) into the low half of an xmm.
            let raw = _mm_set1_epi64x(*i8_ptr);
            // Widen i8 -> i32 (8 lanes).
            let widened = _mm256_cvtepi8_epi32(raw);
            let f = _mm256_cvtepi32_ps(widened);
            let qv = _mm256_loadu_ps(q.as_ptr().add(i * 8));
            acc = _mm256_fmadd_ps(qv, f, acc);
        }
        let mut tail = 0.0f32;
        for i in (chunks * 8)..dim {
            tail += q[i] * (row[i] as f32);
        }
        let mut buf = [0.0f32; 8];
        _mm256_storeu_ps(buf.as_mut_ptr(), acc);
        buf.iter().sum::<f32>() + tail
    }
}

#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
unsafe fn dot_f32_i8_neon(q: &[f32], row: &[i8]) -> f32 {
    unsafe {
        use std::arch::aarch64::*;
        let dim = q.len();
        let mut acc = vdupq_n_f32(0.0);
        // Process 8 lanes per step (NEON's i8x8 widens cleanly to i16x8 then
        // i32x4 + i32x4, then to f32x4 + f32x4).
        let chunks = dim / 8;
        for i in 0..chunks {
            let i8x8 = vld1_s8(row.as_ptr().add(i * 8));
            let i16x8 = vmovl_s8(i8x8);
            let i32_lo = vmovl_s16(vget_low_s16(i16x8));
            let i32_hi = vmovl_s16(vget_high_s16(i16x8));
            let f_lo = vcvtq_f32_s32(i32_lo);
            let f_hi = vcvtq_f32_s32(i32_hi);
            let q_lo = vld1q_f32(q.as_ptr().add(i * 8));
            let q_hi = vld1q_f32(q.as_ptr().add(i * 8 + 4));
            acc = vfmaq_f32(acc, q_lo, f_lo);
            acc = vfmaq_f32(acc, q_hi, f_hi);
        }
        let mut tail = 0.0f32;
        for i in (chunks * 8)..dim {
            tail += q[i] * (row[i] as f32);
        }
        vaddvq_f32(acc) + tail
    }
}

// ---------------------------------------------------------------------------
// Int8 batch helper
// ---------------------------------------------------------------------------

/// Score every row of an int8 embeddings section against `q`.
/// `out[i]` is the cosine score; the runtime sorts these.
pub fn score_int8_section(q: &[f32], view: &Int8EmbeddingsView<'_>, out: &mut [f32]) {
    debug_assert_eq!(out.len(), view.n);
    for (i, slot) in out.iter_mut().enumerate().take(view.n) {
        let scale = view.scale(i);
        let row = view.row(i);
        *slot = dot_f32_i8(q, row, scale);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn random_normalized(seed: u64, dim: usize) -> Vec<f32> {
        // Linear congruential — deterministic, no rand dep needed.
        let mut state = seed
            .wrapping_mul(2862933555777941757)
            .wrapping_add(3037000493);
        let mut v = Vec::with_capacity(dim);
        for _ in 0..dim {
            state = state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            let bits = (state >> 32) as u32;
            let u = (bits as f32) / (u32::MAX as f32);
            v.push(u - 0.5);
        }
        let n = v.iter().map(|x| x * x).sum::<f32>().sqrt();
        if n > 0.0 {
            for x in &mut v {
                *x /= n;
            }
        }
        v
    }

    fn f32_to_le_bytes(v: &[f32]) -> Vec<u8> {
        let mut out = Vec::with_capacity(v.len() * 4);
        for x in v {
            out.extend_from_slice(&x.to_le_bytes());
        }
        out
    }

    #[test]
    fn detect_returns_a_backend() {
        let b = detect_backend();
        // Cached after first call.
        assert_eq!(b, detect_backend());
    }

    #[test]
    fn f32_simd_matches_scalar() {
        for dim in [4, 7, 16, 31, 64, 384] {
            let q = random_normalized(0xDEAD_BEEF, dim);
            let row = random_normalized(0xC0DE_FACE, dim);
            let row_bytes = f32_to_le_bytes(&row);
            let scalar = dot_f32_scalar(&q, &row_bytes);
            let dispatched = dot_f32_bytes(&q, &row_bytes);
            assert!(
                (scalar - dispatched).abs() < 1e-4,
                "dim={} scalar={} dispatched={} diff={}",
                dim,
                scalar,
                dispatched,
                (scalar - dispatched).abs()
            );
        }
    }

    #[test]
    fn f16_simd_matches_scalar() {
        for dim in [4, 8, 31, 64, 384] {
            let q = random_normalized(0x1234, dim);
            let row = random_normalized(0x5678, dim);
            let f16_bytes = nest_format::f32_to_f16_bytes(&row);
            let scalar = dot_f32_f16_scalar(&q, &f16_bytes);
            let dispatched = dot_f32_f16_bytes(&q, &f16_bytes);
            assert!(
                (scalar - dispatched).abs() < 1e-3,
                "dim={} scalar={} dispatched={}",
                dim,
                scalar,
                dispatched
            );
        }
    }

    #[test]
    fn i8_simd_matches_scalar() {
        for dim in [8, 31, 64, 384] {
            let q = random_normalized(0xABCD, dim);
            let row = random_normalized(0x4321, dim);
            let (scale, q_row) = nest_format::quantize_f32_to_i8(&row);
            let scalar = dot_f32_i8_scalar(&q, &q_row) * scale;
            let dispatched = dot_f32_i8(&q, &q_row, scale);
            assert!(
                (scalar - dispatched).abs() < 1e-4,
                "dim={} scalar={} dispatched={} diff={}",
                dim,
                scalar,
                dispatched,
                (scalar - dispatched).abs()
            );
        }
    }
}
