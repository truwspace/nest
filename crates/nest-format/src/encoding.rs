//! Section payload encoding (raw / zstd / float16 / int8).
//!
//! Two orthogonal axes:
//!
//! - **Wire encoding** of a section payload (`SECTION_ENCODING_*` in
//!   `layout`): how the bytes are stored on disk. `raw` and `zstd`
//!   apply to any non-embedding section. `float16` and `int8` only
//!   apply to the embeddings section.
//!
//! - **Logical dtype** of an embeddings section (`Manifest::dtype`):
//!   how to interpret the bytes after wire decoding. `float32` is the
//!   recall-max baseline; `float16` and `int8` are smaller-but-lossy
//!   variants that always accumulate in `f32` at search time.
//!
//! Section checksums (`SectionEntry::checksum`) hash the **physical**
//! bytes as stored. `content_hash` hashes the **decoded** bytes so two
//! files with the same logical content but different wire encoding
//! still produce the same content_hash for non-quantized sections.

use crate::error::NestError;
use crate::layout::{
    SECTION_ENCODING_FLOAT16, SECTION_ENCODING_INT8, SECTION_ENCODING_RAW, SECTION_ENCODING_ZSTD,
};
use std::borrow::Cow;

/// Quantization scale layout for `int8` embeddings.
///
/// On disk:
/// ```text
///   u32 LE  payload_version = 1
///   u32 LE  scale_kind      = 0  (per-vector f32)
///   f32 LE * n              scales (one per vector)
///   i8     * (n * dim)      quantized embeddings, row-major
/// ```
///
/// The scale is the multiplier such that `f32_value ≈ i8_value * scale`.
/// We pick per-vector scales so a single outlier vector cannot crush
/// resolution for the whole corpus.
pub const INT8_PAYLOAD_VERSION: u32 = 1;
pub const INT8_SCALE_KIND_PER_VECTOR: u32 = 0;
pub const INT8_PREFIX_SIZE: usize = 8;

/// Default zstd compression level. 19 is in the "high" tier — slow to
/// encode but a one-time cost and yields ~30% smaller text payloads
/// than the default level 3.
pub const DEFAULT_ZSTD_LEVEL: i32 = 19;

/// Decode a section payload from its on-disk encoding to the logical
/// bytes a reader consumes. For `raw` this is a borrow; for `zstd` it
/// is an owned decompressed buffer.
///
/// Embedding-only encodings (`float16`, `int8`) are returned as-is —
/// their physical bytes ARE their canonical representation. The runtime
/// dispatches on `dtype` to interpret them.
pub fn decode_payload<'a>(encoding: u32, bytes: &'a [u8]) -> crate::Result<Cow<'a, [u8]>> {
    match encoding {
        SECTION_ENCODING_RAW | SECTION_ENCODING_FLOAT16 | SECTION_ENCODING_INT8 => {
            Ok(Cow::Borrowed(bytes))
        }
        SECTION_ENCODING_ZSTD => {
            let out = zstd::decode_all(bytes).map_err(|e| NestError::MalformedSectionPayload {
                section_id: 0,
                reason: format!("zstd decompression failed: {}", e),
            })?;
            Ok(Cow::Owned(out))
        }
        other => Err(NestError::UnsupportedSectionEncoding {
            section_id: 0,
            encoding: other,
        }),
    }
}

/// Compress with zstd at `DEFAULT_ZSTD_LEVEL`. Returns the compressed
/// bytes ready to write as the section payload.
pub fn zstd_encode(bytes: &[u8]) -> crate::Result<Vec<u8>> {
    zstd::encode_all(bytes, DEFAULT_ZSTD_LEVEL)
        .map_err(|e| NestError::InvalidInput(format!("zstd compression failed: {}", e)))
}

/// Convert an L2-normalized float32 embedding to float16, returning the
/// little-endian byte representation. Not a checked NaN/Inf path —
/// callers must validate inputs upstream.
pub fn f32_to_f16_bytes(values: &[f32]) -> Vec<u8> {
    let mut out = Vec::with_capacity(values.len() * 2);
    for &v in values {
        let half = half::f16::from_f32(v);
        out.extend_from_slice(&half.to_le_bytes());
    }
    out
}

/// Decode a float16 byte slice into f32 values, accumulating in f32
/// for downstream dot-product accuracy.
#[inline]
pub fn f16_bytes_to_f32(bytes: &[u8]) -> Vec<f32> {
    let mut out = Vec::with_capacity(bytes.len() / 2);
    for chunk in bytes.chunks_exact(2) {
        let h = half::f16::from_le_bytes([chunk[0], chunk[1]]);
        out.push(h.to_f32());
    }
    out
}

/// Quantize an L2-normalized float32 embedding to int8 with a per-vector
/// scale. Returns `(scale, i8_bytes)` where `f32_value ≈ i8 * scale`.
///
/// L2-normalized vectors live in `[-1, 1]`; in practice the largest
/// component is well below 1 so we map `max(|v|)` to 127 to use the
/// full int8 range. Re-normalization at query time accumulates in f32.
pub fn quantize_f32_to_i8(values: &[f32]) -> (f32, Vec<i8>) {
    let max_abs = values.iter().fold(0.0f32, |acc, &v| acc.max(v.abs()));
    if max_abs == 0.0 {
        // Pathological zero vector — quantize to all zeros with scale 1.
        // The reader's zero-norm guard will reject queries against this.
        return (1.0, vec![0i8; values.len()]);
    }
    let scale = max_abs / 127.0;
    let inv_scale = 1.0 / scale;
    let q: Vec<i8> = values
        .iter()
        .map(|&v| {
            let scaled = (v * inv_scale).round();
            scaled.clamp(-127.0, 127.0) as i8
        })
        .collect();
    (scale, q)
}

/// Encode the int8 embeddings section payload. Layout matches
/// `INT8_PAYLOAD_VERSION` / `INT8_SCALE_KIND_PER_VECTOR`.
///
/// `embeddings` is `n * dim` row-major f32 values. Returns a buffer
/// ready to write as the embeddings section (encoding = INT8).
pub fn encode_int8_embeddings(embeddings: &[f32], n: usize, dim: usize) -> crate::Result<Vec<u8>> {
    if embeddings.len() != n * dim {
        return Err(NestError::InvalidInput(format!(
            "encode_int8_embeddings: got {} f32 values for n={} dim={}",
            embeddings.len(),
            n,
            dim
        )));
    }
    let mut out = Vec::with_capacity(INT8_PREFIX_SIZE + n * 4 + n * dim);
    out.extend_from_slice(&INT8_PAYLOAD_VERSION.to_le_bytes());
    out.extend_from_slice(&INT8_SCALE_KIND_PER_VECTOR.to_le_bytes());
    let mut scales: Vec<u8> = Vec::with_capacity(n * 4);
    let mut bodies: Vec<u8> = Vec::with_capacity(n * dim);
    for i in 0..n {
        let row = &embeddings[i * dim..(i + 1) * dim];
        let (scale, q) = quantize_f32_to_i8(row);
        scales.extend_from_slice(&scale.to_le_bytes());
        // i8 -> u8 is a bitcast (two's complement preserved).
        bodies.extend(q.iter().map(|&v| v as u8));
    }
    out.extend_from_slice(&scales);
    out.extend_from_slice(&bodies);
    Ok(out)
}

/// Decoded view over an int8 embeddings payload. The slices borrow
/// from the input bytes (no copy).
pub struct Int8EmbeddingsView<'a> {
    pub scales: &'a [u8], // n * 4 bytes (f32 LE)
    pub bodies: &'a [u8], // n * dim bytes (i8, bitcast to u8)
    pub n: usize,
    pub dim: usize,
}

impl<'a> Int8EmbeddingsView<'a> {
    pub fn parse(bytes: &'a [u8], n: usize, dim: usize) -> crate::Result<Self> {
        let want = INT8_PREFIX_SIZE + n * 4 + n * dim;
        if bytes.len() != want {
            return Err(NestError::EmbeddingSizeMismatch {
                expected: want,
                got: bytes.len(),
            });
        }
        let version = u32::from_le_bytes(bytes[0..4].try_into().unwrap());
        if version != INT8_PAYLOAD_VERSION {
            return Err(NestError::UnsupportedSectionVersion {
                section_id: crate::layout::SECTION_EMBEDDINGS,
                version,
            });
        }
        let kind = u32::from_le_bytes(bytes[4..8].try_into().unwrap());
        if kind != INT8_SCALE_KIND_PER_VECTOR {
            return Err(NestError::MalformedSectionPayload {
                section_id: crate::layout::SECTION_EMBEDDINGS,
                reason: format!("int8 scale_kind {} not supported", kind),
            });
        }
        let scales_end = INT8_PREFIX_SIZE + n * 4;
        Ok(Self {
            scales: &bytes[INT8_PREFIX_SIZE..scales_end],
            bodies: &bytes[scales_end..],
            n,
            dim,
        })
    }

    /// Read scale[i] as f32.
    #[inline]
    pub fn scale(&self, i: usize) -> f32 {
        let off = i * 4;
        f32::from_le_bytes([
            self.scales[off],
            self.scales[off + 1],
            self.scales[off + 2],
            self.scales[off + 3],
        ])
    }

    /// Borrow row[i] as `&[i8]`.
    #[inline]
    pub fn row(&self, i: usize) -> &'a [i8] {
        let start = i * self.dim;
        let end = start + self.dim;
        // i8 has the same layout as u8 (one-byte signed).
        unsafe {
            std::slice::from_raw_parts(self.bodies[start..end].as_ptr() as *const i8, self.dim)
        }
    }
}

/// Expected size of the embeddings section for a given dtype. Returns
/// `None` for dtypes where the size depends on a payload prefix (int8).
pub fn expected_embeddings_size(dtype: &str, n: usize, dim: usize) -> Option<usize> {
    match dtype {
        "float32" => Some(n * dim * 4),
        "float16" => Some(n * dim * 2),
        "int8" => Some(INT8_PREFIX_SIZE + n * 4 + n * dim),
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn zstd_roundtrip_preserves_bytes() {
        let original = b"hello hello hello world world world".repeat(64);
        let compressed = zstd_encode(&original).unwrap();
        assert!(
            compressed.len() < original.len(),
            "zstd should shrink repetitive text"
        );
        let decoded = decode_payload(SECTION_ENCODING_ZSTD, &compressed).unwrap();
        assert_eq!(decoded.as_ref(), original.as_slice());
    }

    #[test]
    fn raw_decode_borrows() {
        let bytes = b"plain";
        let decoded = decode_payload(SECTION_ENCODING_RAW, bytes).unwrap();
        assert!(matches!(decoded, std::borrow::Cow::Borrowed(_)));
    }

    #[test]
    fn unknown_encoding_rejected() {
        let res = decode_payload(99, &[]);
        assert!(matches!(
            res,
            Err(NestError::UnsupportedSectionEncoding { encoding: 99, .. })
        ));
    }

    #[test]
    fn f16_roundtrip_within_tolerance() {
        let v: Vec<f32> = (0..16).map(|i| (i as f32) * 0.05).collect();
        let bytes = f32_to_f16_bytes(&v);
        let back = f16_bytes_to_f32(&bytes);
        assert_eq!(back.len(), v.len());
        for (a, b) in v.iter().zip(back.iter()) {
            // f16 has ~1e-3 relative precision in this range.
            assert!((a - b).abs() < 1e-3, "{} vs {}", a, b);
        }
    }

    #[test]
    fn int8_quantize_and_dequantize() {
        let v: Vec<f32> = vec![1.0, -1.0, 0.5, -0.5, 0.0, 0.25];
        let (scale, q) = quantize_f32_to_i8(&v);
        assert!(scale > 0.0);
        // Largest absolute component should map close to ±127.
        assert!(q.iter().any(|&x| x == 127 || x == -127));
        // Reconstructed values should be within scale tolerance.
        for (orig, &qi) in v.iter().zip(q.iter()) {
            let recon = qi as f32 * scale;
            assert!((orig - recon).abs() <= scale * 1.01);
        }
    }

    #[test]
    fn int8_section_roundtrip() {
        let n = 4;
        let dim = 8;
        let mut emb: Vec<f32> = Vec::with_capacity(n * dim);
        for i in 0..n {
            let mut v = vec![0.0f32; dim];
            v[i % dim] = 1.0;
            // L2-normalized basis vector
            emb.extend_from_slice(&v);
        }
        let payload = encode_int8_embeddings(&emb, n, dim).unwrap();
        let view = Int8EmbeddingsView::parse(&payload, n, dim).unwrap();
        assert_eq!(view.n, n);
        assert_eq!(view.dim, dim);
        for i in 0..n {
            let scale = view.scale(i);
            let row = view.row(i);
            assert_eq!(row.len(), dim);
            // Reconstructed row should be near the original.
            let recon: Vec<f32> = row.iter().map(|&x| x as f32 * scale).collect();
            for (orig, r) in emb[i * dim..(i + 1) * dim].iter().zip(recon.iter()) {
                assert!((orig - r).abs() < 0.02, "{} vs {}", orig, r);
            }
        }
    }
}
