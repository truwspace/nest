use thiserror::Error;

#[derive(Debug, Error)]
pub enum NestError {
    #[error("I/O error: {0}")]
    Io(#[from] std::io::Error),
    #[error("magic mismatch: expected {expected:?}, got {got:?}")]
    MagicMismatch { expected: [u8; 4], got: [u8; 4] },
    #[error("unsupported version: {0}.{1}")]
    UnsupportedVersion(u16, u16),
    #[error("invalid header checksum")]
    InvalidHeaderChecksum,
    #[error("section {0} offset out of bounds: {1}")]
    SectionOffsetOutOfBounds(u32, u64),
    #[error("section {0} checksum mismatch")]
    SectionChecksumMismatch(u32),
    #[error("section {0} not found")]
    SectionNotFound(u32),
    #[error("footer hash mismatch")]
    FooterHashMismatch,
    #[error("file size mismatch: expected {expected}, got {got}")]
    FileSizeMismatch { expected: u64, got: u64 },
    #[error("invalid chunk count: {0}")]
    InvalidChunkCount(u32),
    #[error("invalid embedding dimension: {0}")]
    InvalidEmbeddingDimension(u32),
    #[error("NaN or Inf detected in embeddings")]
    InvalidEmbeddingValue,
    #[error("JSON error: {0}")]
    Json(#[from] serde_json::Error),
    #[error("manifest validation failed: {0}")]
    ManifestValidation(String),
    #[error("query validation failed: {0}")]
    QueryValidation(String),
    #[error("dimension mismatch: expected {expected}, got {got}")]
    DimensionMismatch { expected: u32, got: u32 },
    #[error("invalid k parameter: {0}")]
    InvalidK(i32),
    #[error("empty query")]
    EmptyQuery,
    #[error("file not found: {0}")]
    FileNotFound(String),
    #[error("unexpected end of file")]
    UnexpectedEof,
    #[error("unsupported feature: {0}")]
    UnsupportedFeature(String),
}

pub type Result<T> = std::result::Result<T, NestError>;
