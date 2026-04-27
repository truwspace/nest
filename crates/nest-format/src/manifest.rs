use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;

/// Canonical JCS-serialized manifest for a .nest file.
/// We keep keys ordered and deterministic by using BTreeMap for extras.
#[derive(Clone, Debug, Default, Serialize, Deserialize, PartialEq)]
pub struct Manifest {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub title: Option<String>,
    pub embedding_model: String,
    pub embedding_dim: u32,
    pub n_chunks: u64,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub version: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub created: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub description: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub authors: Option<Vec<String>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub license: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub provenance: Option<Vec<ProvenanceEntry>>,
    #[serde(flatten)]
    pub extra: BTreeMap<String, serde_json::Value>,
}

impl Manifest {
    pub fn validate(&self) -> crate::Result<()> {
        use crate::error::NestError;
        if self.embedding_model.is_empty() {
            return Err(NestError::ManifestValidation(
                "embedding_model must not be empty".into(),
            ));
        }
        if self.embedding_dim == 0 {
            return Err(NestError::ManifestValidation(
                "embedding_dim must be > 0".into(),
            ));
        }
        Ok(())
    }

    /// Serialize to canonical JSON (sorted keys) without extra whitespace.
    pub fn to_canonical_json(&self) -> crate::Result<Vec<u8>> {
        let mut buf = Vec::new();
        let ser = serde_json::Serializer::with_formatter(
            &mut buf,
            serde_json::ser::CompactFormatter,
        );
        self.serialize(ser).map_err(crate::NestError::Json)?;
        Ok(buf)
    }
}

#[derive(Clone, Debug, Default, Serialize, Deserialize, PartialEq)]
pub struct ProvenanceEntry {
    pub source_uri: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub ingestion_date: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub processor: Option<String>,
    #[serde(flatten)]
    pub extra: BTreeMap<String, serde_json::Value>,
}
