//! Decode the embeddings section into a flat row-major f32 buffer
//! regardless of the on-disk dtype. Used by the ANN index since the
//! HNSW graph accumulates distances in f32. Cost: one allocation of
//! `n * dim * 4` bytes — the price of being dtype-agnostic.

use nest_format::{Int8EmbeddingsView, NestError};

use crate::error::RuntimeError;

pub(crate) fn materialize_f32_vectors(
    dtype: &str,
    bytes: &[u8],
    n: usize,
    dim: usize,
) -> Result<Vec<f32>, RuntimeError> {
    match dtype {
        "float32" => {
            let mut out = vec![0.0f32; n * dim];
            for (i, slot) in out.iter_mut().enumerate() {
                let off = i * 4;
                *slot = f32::from_le_bytes([
                    bytes[off],
                    bytes[off + 1],
                    bytes[off + 2],
                    bytes[off + 3],
                ]);
            }
            Ok(out)
        }
        "float16" => Ok(nest_format::f16_bytes_to_f32(bytes)),
        "int8" => {
            let view = Int8EmbeddingsView::parse(bytes, n, dim).map_err(RuntimeError::Format)?;
            let mut out = vec![0.0f32; n * dim];
            for i in 0..n {
                let scale = view.scale(i);
                let row = view.row(i);
                for j in 0..dim {
                    out[i * dim + j] = row[j] as f32 * scale;
                }
            }
            Ok(out)
        }
        other => Err(RuntimeError::Format(NestError::UnsupportedDType(
            other.into(),
        ))),
    }
}
