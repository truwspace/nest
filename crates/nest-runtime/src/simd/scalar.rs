//! Scalar fallback dot products. Auto-vectorizes well in release mode
//! and is the path used when SIMD detection fails or
//! `NEST_FORCE_SCALAR=1`. Also the reference implementation the SIMD
//! parity tests compare against.

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

#[inline]
pub fn dot_f32_i8_scalar(q: &[f32], row: &[i8]) -> f32 {
    let mut acc = 0.0f32;
    for (qv, &iv) in q.iter().zip(row.iter()) {
        acc += *qv * (iv as f32);
    }
    acc
}
