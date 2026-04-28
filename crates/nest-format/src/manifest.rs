use crate::error::NestError;
use crate::layout::{NEST_FORMAT_VERSION, NEST_SCHEMA_VERSION};
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;

/// Default embedding dtype; v1.0 ships float32 as the recall-max baseline.
/// Compressed/quantized presets declare `dtype = "float16"` or `"int8"`.
pub const SUPPORTED_DTYPE: &str = "float32";
pub const SUPPORTED_METRIC: &str = "ip";
pub const SUPPORTED_SCORE_TYPE: &str = "cosine";
pub const SUPPORTED_NORMALIZE: &str = "l2";
pub const SUPPORTED_INDEX_TYPE: &str = "exact";
pub const SUPPORTED_RERANK_POLICY: &str = "none";

/// Every dtype the reader understands for the embeddings section.
pub const ALLOWED_DTYPES: &[&str] = &["float32", "float16", "int8"];
/// Every index_type the reader understands for the search path.
pub const ALLOWED_INDEX_TYPES: &[&str] = &["exact", "hnsw", "hybrid"];
/// Every rerank policy the reader understands. `exact` means an ANN/BM25
/// candidate set is rescored with real cosine before returning.
pub const ALLOWED_RERANK_POLICIES: &[&str] = &["none", "exact"];
/// Every score_type the reader understands.
pub const ALLOWED_SCORE_TYPES: &[&str] = &["cosine", "hybrid_rrf"];

/// Capabilities advertised by a .nest file. v1 only requires `supports_exact`
/// and `supports_reproducible_build` to be true; the rest are forward-looking
/// flags so a runtime can decide what to do without reading every section.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Capabilities {
    pub supports_exact: bool,
    pub supports_ann: bool,
    pub supports_bm25: bool,
    pub supports_citations: bool,
    pub supports_reproducible_build: bool,
}

impl Default for Capabilities {
    fn default() -> Self {
        Self {
            supports_exact: true,
            supports_ann: false,
            supports_bm25: false,
            supports_citations: true,
            supports_reproducible_build: true,
        }
    }
}

/// JCS-canonical manifest for a .nest file.
///
/// Field order on disk follows declaration order. Extra keys land in `extra`
/// and are serialized in BTreeMap order; this keeps the JSON deterministic.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct Manifest {
    pub format_version: u32,
    pub schema_version: u32,
    pub embedding_model: String,
    pub embedding_dim: u32,
    pub n_chunks: u64,
    pub dtype: String,
    pub metric: String,
    pub score_type: String,
    pub normalize: String,
    pub index_type: String,
    pub rerank_policy: String,
    pub model_hash: String,
    pub chunker_version: String,
    pub capabilities: Capabilities,

    #[serde(skip_serializing_if = "Option::is_none")]
    pub title: Option<String>,
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

    #[serde(flatten)]
    pub extra: BTreeMap<String, serde_json::Value>,
}

impl Default for Manifest {
    fn default() -> Self {
        Self {
            format_version: NEST_FORMAT_VERSION,
            schema_version: NEST_SCHEMA_VERSION,
            embedding_model: String::new(),
            embedding_dim: 0,
            n_chunks: 0,
            dtype: SUPPORTED_DTYPE.to_string(),
            metric: SUPPORTED_METRIC.to_string(),
            score_type: SUPPORTED_SCORE_TYPE.to_string(),
            normalize: SUPPORTED_NORMALIZE.to_string(),
            index_type: SUPPORTED_INDEX_TYPE.to_string(),
            rerank_policy: SUPPORTED_RERANK_POLICY.to_string(),
            model_hash: String::new(),
            chunker_version: String::new(),
            capabilities: Capabilities::default(),
            title: None,
            version: None,
            created: None,
            description: None,
            authors: None,
            license: None,
            extra: BTreeMap::new(),
        }
    }
}

impl Manifest {
    /// Validate that this manifest matches the v1 contract. Reject any
    /// unsupported value with a typed error rather than a string blob.
    ///
    /// Version skew policy:
    ///   - `format_version` > reader's supported version → reject (the
    ///     binary container may have changed in incompatible ways).
    ///   - `schema_version` > reader's supported version → reject (new
    ///     required fields the reader cannot enforce).
    ///   - Lower or equal versions are accepted, on the assumption that
    ///     the reader knows how to interpret older manifests.
    pub fn validate(&self) -> crate::Result<()> {
        if self.format_version > NEST_FORMAT_VERSION {
            return Err(NestError::UnsupportedFormatVersion(self.format_version));
        }
        if self.schema_version > NEST_SCHEMA_VERSION {
            return Err(NestError::UnsupportedSchemaVersion(self.schema_version));
        }
        if self.embedding_model.is_empty() {
            return Err(NestError::ManifestInvalid(
                "embedding_model must not be empty".into(),
            ));
        }
        if self.embedding_dim == 0 {
            return Err(NestError::ManifestInvalid(
                "embedding_dim must be > 0".into(),
            ));
        }
        if self.n_chunks == 0 {
            return Err(NestError::ManifestInvalid("n_chunks must be > 0".into()));
        }
        if self.chunker_version.is_empty() {
            return Err(NestError::ManifestInvalid(
                "chunker_version must not be empty".into(),
            ));
        }
        validate_model_hash(&self.model_hash)?;

        if !ALLOWED_DTYPES.contains(&self.dtype.as_str()) {
            return Err(NestError::UnsupportedDType(self.dtype.clone()));
        }
        if self.metric != SUPPORTED_METRIC {
            return Err(NestError::UnsupportedMetric(self.metric.clone()));
        }
        if !ALLOWED_SCORE_TYPES.contains(&self.score_type.as_str()) {
            return Err(NestError::UnsupportedScoreType(self.score_type.clone()));
        }
        if self.normalize != SUPPORTED_NORMALIZE {
            return Err(NestError::UnsupportedNormalize(self.normalize.clone()));
        }
        if !ALLOWED_INDEX_TYPES.contains(&self.index_type.as_str()) {
            return Err(NestError::UnsupportedIndexType(self.index_type.clone()));
        }
        if !ALLOWED_RERANK_POLICIES.contains(&self.rerank_policy.as_str()) {
            return Err(NestError::UnsupportedRerankPolicy(
                self.rerank_policy.clone(),
            ));
        }
        // ANN/hybrid index_types must declare an exact rerank so the final
        // score remains the real cosine value the user can trust.
        if (self.index_type == "hnsw" || self.index_type == "hybrid")
            && self.rerank_policy != "exact"
        {
            return Err(NestError::ManifestInvalid(format!(
                "index_type={} requires rerank_policy=\"exact\"",
                self.index_type
            )));
        }
        if !self.capabilities.supports_exact {
            return Err(NestError::ManifestInvalid(
                "capabilities.supports_exact must be true (exact path is the ground truth)".into(),
            ));
        }
        if !self.capabilities.supports_reproducible_build {
            return Err(NestError::ManifestInvalid(
                "capabilities.supports_reproducible_build must be true".into(),
            ));
        }
        if self.index_type == "hnsw" && !self.capabilities.supports_ann {
            return Err(NestError::ManifestInvalid(
                "index_type=hnsw requires capabilities.supports_ann=true".into(),
            ));
        }
        if self.index_type == "hybrid" && !self.capabilities.supports_bm25 {
            return Err(NestError::ManifestInvalid(
                "index_type=hybrid requires capabilities.supports_bm25=true".into(),
            ));
        }
        Ok(())
    }

    /// Serialize the manifest to canonical JSON: declaration order for the
    /// known fields, BTreeMap order for `extra`, no whitespace.
    pub fn to_canonical_json(&self) -> crate::Result<Vec<u8>> {
        let mut buf = Vec::new();
        let mut ser =
            serde_json::Serializer::with_formatter(&mut buf, serde_json::ser::CompactFormatter);
        self.serialize(&mut ser).map_err(NestError::Json)?;
        Ok(buf)
    }
}

/// A model_hash must be of the form `sha256:<64 hex chars>`. Anything else
/// is a contract violation: we want every claim about provenance to be
/// machine-verifiable.
fn validate_model_hash(s: &str) -> crate::Result<()> {
    let rest = s.strip_prefix("sha256:").ok_or_else(|| {
        NestError::InvalidModelHash(format!("expected 'sha256:<hex>' prefix, got '{}'", s))
    })?;
    if rest.len() != 64 || !rest.chars().all(|c| c.is_ascii_hexdigit()) {
        return Err(NestError::InvalidModelHash(format!(
            "expected 64 hex chars after 'sha256:', got '{}'",
            rest
        )));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn good_manifest() -> Manifest {
        Manifest {
            embedding_model: "demo".into(),
            embedding_dim: 4,
            n_chunks: 1,
            chunker_version: "demo-chunker/1".into(),
            model_hash: format!("sha256:{}", "0".repeat(64)),
            ..Default::default()
        }
    }

    #[test]
    fn default_is_invalid_until_filled() {
        assert!(Manifest::default().validate().is_err());
    }

    #[test]
    fn valid_manifest_passes() {
        good_manifest().validate().unwrap();
    }

    #[test]
    fn rejects_future_format_version() {
        let mut m = good_manifest();
        m.format_version = NEST_FORMAT_VERSION + 1;
        assert!(matches!(
            m.validate(),
            Err(NestError::UnsupportedFormatVersion(_))
        ));
    }

    #[test]
    fn accepts_older_format_version() {
        // Writer has not yet bumped format_version, so we cannot exercise
        // a true skew here without a fake. As soon as a v2 ships, the
        // assertion below remains a v1 reader's contract: smaller versions
        // are accepted (assuming required field set is unchanged).
        let mut m = good_manifest();
        if NEST_FORMAT_VERSION > 0 {
            m.format_version = NEST_FORMAT_VERSION - 1;
            assert!(m.validate().is_ok());
        }
    }

    #[test]
    fn rejects_future_schema_version() {
        let mut m = good_manifest();
        m.schema_version = NEST_SCHEMA_VERSION + 1;
        assert!(matches!(
            m.validate(),
            Err(NestError::UnsupportedSchemaVersion(_))
        ));
    }

    #[test]
    fn rejects_unsupported_dtype() {
        let mut m = good_manifest();
        m.dtype = "bfloat16".into();
        assert!(matches!(m.validate(), Err(NestError::UnsupportedDType(_))));
    }

    #[test]
    fn accepts_float16_and_int8_dtypes() {
        for dt in ["float32", "float16", "int8"] {
            let mut m = good_manifest();
            m.dtype = dt.into();
            m.validate()
                .unwrap_or_else(|e| panic!("{} rejected: {}", dt, e));
        }
    }

    #[test]
    fn hnsw_requires_exact_rerank() {
        let mut m = good_manifest();
        m.index_type = "hnsw".into();
        m.rerank_policy = "none".into();
        m.capabilities.supports_ann = true;
        assert!(matches!(m.validate(), Err(NestError::ManifestInvalid(_))));
        m.rerank_policy = "exact".into();
        m.validate().unwrap();
    }

    #[test]
    fn hybrid_requires_bm25_capability() {
        let mut m = good_manifest();
        m.index_type = "hybrid".into();
        m.rerank_policy = "exact".into();
        m.score_type = "hybrid_rrf".into();
        m.capabilities.supports_bm25 = false;
        assert!(matches!(m.validate(), Err(NestError::ManifestInvalid(_))));
        m.capabilities.supports_bm25 = true;
        m.validate().unwrap();
    }

    #[test]
    fn rejects_unsupported_metric() {
        let mut m = good_manifest();
        m.metric = "l2".into();
        assert!(matches!(m.validate(), Err(NestError::UnsupportedMetric(_))));
    }

    #[test]
    fn rejects_invalid_model_hash() {
        let mut m = good_manifest();
        m.model_hash = "deadbeef".into();
        assert!(matches!(m.validate(), Err(NestError::InvalidModelHash(_))));
    }

    #[test]
    fn rejects_missing_chunker_version() {
        let mut m = good_manifest();
        m.chunker_version = "".into();
        assert!(matches!(m.validate(), Err(NestError::ManifestInvalid(_))));
    }

    #[test]
    fn capabilities_required_for_v1() {
        let mut m = good_manifest();
        m.capabilities.supports_exact = false;
        assert!(matches!(m.validate(), Err(NestError::ManifestInvalid(_))));
    }

    #[test]
    fn canonical_json_is_deterministic() {
        let m1 = good_manifest();
        let m2 = good_manifest();
        assert_eq!(
            m1.to_canonical_json().unwrap(),
            m2.to_canonical_json().unwrap()
        );
    }
}
