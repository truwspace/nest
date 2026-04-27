use thiserror::Error;

#[derive(Debug, Error)]
pub enum RuntimeError {
    #[error(transparent)]
    Format(#[from] nest_format::NestError),
    #[error("{0}")]
    Io(#[from] std::io::Error),
    #[error("dimension mismatch: expected {expected}, got {got}")]
    DimensionMismatch { expected: usize, got: usize },
    #[error("invalid k: {0}")]
    InvalidK(i32),
    #[error("empty query")]
    EmptyQuery,
    #[error("zero-norm query")]
    ZeroNormQuery,
    #[error("NaN or Inf in query")]
    InvalidQueryValue,
}
