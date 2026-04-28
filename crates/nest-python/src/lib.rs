use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList};

#[pyclass]
struct NestFile {
    rt: nest_runtime::MmapNestFile,
}

#[pymethods]
impl NestFile {
    #[staticmethod]
    fn open(path: &str) -> PyResult<Self> {
        let rt = nest_runtime::MmapNestFile::open(std::path::Path::new(path))
            .map_err(|e| PyValueError::new_err(format!("{}", e)))?;
        Ok(Self { rt })
    }

    fn search(&self, query: &Bound<PyAny>, k: i32) -> PyResult<Vec<SearchHitPy>> {
        let qvec: Vec<f32> = query
            .extract()
            .map_err(|e| PyValueError::new_err(format!("invalid query vector: {}", e)))?;
        let res = self
            .rt
            .search(&qvec, k)
            .map_err(|e| PyValueError::new_err(format!("{}", e)))?;
        Ok(res.hits.into_iter().map(SearchHitPy::from).collect())
    }

    /// HNSW ANN search with exact rerank. Falls back to `search()` if
    /// the file has no HNSW section.
    fn search_ann(&self, query: &Bound<PyAny>, k: i32, ef: usize) -> PyResult<Vec<SearchHitPy>> {
        let qvec: Vec<f32> = query
            .extract()
            .map_err(|e| PyValueError::new_err(format!("invalid query vector: {}", e)))?;
        let res = self
            .rt
            .search_ann(&qvec, k, ef)
            .map_err(|e| PyValueError::new_err(format!("{}", e)))?;
        Ok(res.hits.into_iter().map(SearchHitPy::from).collect())
    }

    /// Hybrid (BM25 ∪ vector → exact rerank). Falls back to `search()`
    /// when no BM25 section is present.
    fn search_hybrid(
        &self,
        query: &Bound<PyAny>,
        query_text: &str,
        k: i32,
        candidates: usize,
    ) -> PyResult<Vec<SearchHitPy>> {
        let qvec: Vec<f32> = query
            .extract()
            .map_err(|e| PyValueError::new_err(format!("invalid query vector: {}", e)))?;
        let res = self
            .rt
            .search_hybrid(&qvec, query_text, k, candidates)
            .map_err(|e| PyValueError::new_err(format!("{}", e)))?;
        Ok(res.hits.into_iter().map(SearchHitPy::from).collect())
    }

    #[getter]
    fn embedding_dim(&self) -> usize {
        self.rt.embedding_dim()
    }

    #[getter]
    fn n_embeddings(&self) -> usize {
        self.rt.n_embeddings()
    }

    #[getter]
    fn dtype(&self) -> &'static str {
        self.rt.dtype().name()
    }

    #[getter]
    fn simd_backend(&self) -> &'static str {
        self.rt.simd_backend().name()
    }

    #[getter]
    fn has_ann(&self) -> bool {
        self.rt.has_ann()
    }

    #[getter]
    fn has_bm25(&self) -> bool {
        self.rt.has_bm25()
    }

    #[getter]
    fn file_hash(&self) -> String {
        self.rt.file_hash().to_string()
    }

    #[getter]
    fn content_hash(&self) -> String {
        self.rt.content_hash().to_string()
    }

    /// Mirror of `nest inspect`: returns a Python dict with header,
    /// section table, manifest and hashes.
    fn inspect<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        let s = self
            .rt
            .inspect_json()
            .map_err(|e| PyValueError::new_err(format!("{}", e)))?;
        py.import("json")?.call_method1("loads", (s,))
    }

    /// Re-run reader-side validation. Returns `True` on success and
    /// raises `ValueError` (with the reader's typed error in the
    /// message) on any failure.
    fn validate(&self) -> PyResult<bool> {
        self.rt
            .revalidate()
            .map_err(|e| PyValueError::new_err(format!("{}", e)))?;
        Ok(true)
    }
}

#[pyclass(skip_from_py_object)]
#[derive(Clone)]
struct SearchHitPy {
    #[pyo3(get)]
    pub chunk_id: String,
    #[pyo3(get)]
    pub score: f32,
    #[pyo3(get)]
    pub score_type: String,
    #[pyo3(get)]
    pub source_uri: String,
    #[pyo3(get)]
    pub offset_start: u64,
    #[pyo3(get)]
    pub offset_end: u64,
    #[pyo3(get)]
    pub embedding_model: String,
    #[pyo3(get)]
    pub index_type: String,
    #[pyo3(get)]
    pub reranked: bool,
    #[pyo3(get)]
    pub file_hash: String,
    #[pyo3(get)]
    pub content_hash: String,
    #[pyo3(get)]
    pub citation_id: String,
}

impl From<nest_runtime::SearchHit> for SearchHitPy {
    fn from(h: nest_runtime::SearchHit) -> Self {
        Self {
            chunk_id: h.chunk_id,
            score: h.score,
            score_type: h.score_type.to_string(),
            source_uri: h.source_uri,
            offset_start: h.offset_start,
            offset_end: h.offset_end,
            embedding_model: h.embedding_model,
            index_type: h.index_type.to_string(),
            reranked: h.reranked,
            file_hash: h.file_hash,
            content_hash: h.content_hash,
            citation_id: h.citation_id,
        }
    }
}

/// Build a .nest file from already-embedded chunks.
///
/// `chunks` is a list of dicts with keys:
///   - canonical_text: str
///   - source_uri: str
///   - byte_start: int
///   - byte_end: int
///   - embedding: list[float] (length == embedding_dim, L2-normalized)
///
/// Optional manifest fields (`title`, `version`, `created`, `description`,
/// `authors`, `license`) and a free-form `provenance` dict can be passed
/// as kwargs. `reproducible=True` overrides `created` so two builds with
/// identical inputs produce byte-identical output.
///
/// Preset / encoding kwargs:
///   - `preset`: one of "exact" (default), "compressed", "tiny", "hybrid"
///   - `text_encoding`: "raw" | "zstd" (overrides preset)
///   - `dtype`: "float32" | "float16" | "int8" (overrides preset)
///   - `with_hnsw`: bool (overrides preset; default per preset)
///   - `with_bm25`: bool (overrides preset; default per preset)
///   - `hnsw_m`, `hnsw_ef_construction`, `hnsw_seed`: HNSW knobs
#[pyfunction]
#[pyo3(signature = (
    output_path,
    embedding_model,
    embedding_dim,
    chunker_version,
    model_hash,
    chunks,
    *,
    title=None,
    version=None,
    created=None,
    description=None,
    authors=None,
    license=None,
    provenance=None,
    reproducible=false,
    preset="exact",
    text_encoding=None,
    dtype=None,
    with_hnsw=None,
    with_bm25=None,
    hnsw_m=16,
    hnsw_ef_construction=400,
    hnsw_seed=42,
))]
#[allow(clippy::too_many_arguments)]
fn build(
    py: Python<'_>,
    output_path: &str,
    embedding_model: &str,
    embedding_dim: u32,
    chunker_version: &str,
    model_hash: &str,
    chunks: &Bound<PyList>,
    title: Option<String>,
    version: Option<String>,
    created: Option<String>,
    description: Option<String>,
    authors: Option<Vec<String>>,
    license: Option<String>,
    provenance: Option<&Bound<PyDict>>,
    reproducible: bool,
    preset: &str,
    text_encoding: Option<&str>,
    dtype: Option<&str>,
    with_hnsw: Option<bool>,
    with_bm25: Option<bool>,
    hnsw_m: usize,
    hnsw_ef_construction: usize,
    hnsw_seed: u64,
) -> PyResult<String> {
    use nest_format::ChunkInput;
    use nest_format::manifest::Manifest;
    use nest_format::writer::{EmbeddingDType, NestFileBuilder, SectionEncoding};

    let n_chunks = chunks.len() as u64;
    let mut chunk_inputs: Vec<ChunkInput> = Vec::with_capacity(n_chunks as usize);
    for (i, item) in chunks.iter().enumerate() {
        let d: Bound<PyDict> = item
            .cast::<PyDict>()
            .map_err(|_| PyValueError::new_err(format!("chunks[{}] is not a dict", i)))?
            .clone();
        let d = &d;
        let canonical_text: String = d
            .get_item("canonical_text")?
            .ok_or_else(|| PyValueError::new_err(format!("chunks[{}] missing canonical_text", i)))?
            .extract()?;
        let source_uri: String = d
            .get_item("source_uri")?
            .ok_or_else(|| PyValueError::new_err(format!("chunks[{}] missing source_uri", i)))?
            .extract()?;
        let byte_start: u64 = d
            .get_item("byte_start")?
            .ok_or_else(|| PyValueError::new_err(format!("chunks[{}] missing byte_start", i)))?
            .extract()?;
        let byte_end: u64 = d
            .get_item("byte_end")?
            .ok_or_else(|| PyValueError::new_err(format!("chunks[{}] missing byte_end", i)))?
            .extract()?;
        let embedding: Vec<f32> = d
            .get_item("embedding")?
            .ok_or_else(|| PyValueError::new_err(format!("chunks[{}] missing embedding", i)))?
            .extract()?;
        chunk_inputs.push(ChunkInput {
            canonical_text,
            source_uri,
            byte_start,
            byte_end,
            embedding,
        });
    }

    let provenance_value = match provenance {
        Some(p) => {
            let s: String = py.import("json")?.call_method1("dumps", (p,))?.extract()?;
            serde_json::from_str(&s)
                .map_err(|e| PyValueError::new_err(format!("provenance JSON: {}", e)))?
        }
        None => serde_json::json!({}),
    };

    // Resolve preset defaults; explicit kwargs win.
    let (default_text_enc, default_dtype, default_hnsw, default_bm25) = match preset {
        "exact" => (SectionEncoding::Raw, EmbeddingDType::Float32, false, false),
        "compressed" => (SectionEncoding::Zstd, EmbeddingDType::Float16, false, false),
        "tiny" => (SectionEncoding::Zstd, EmbeddingDType::Int8, true, false),
        "hybrid" => (SectionEncoding::Zstd, EmbeddingDType::Float32, true, true),
        other => {
            return Err(PyValueError::new_err(format!(
                "unknown preset: {} (expected exact|compressed|tiny|hybrid)",
                other
            )));
        }
    };
    let text_enc = match text_encoding {
        Some("raw") => SectionEncoding::Raw,
        Some("zstd") => SectionEncoding::Zstd,
        Some(other) => {
            return Err(PyValueError::new_err(format!(
                "unknown text_encoding: {} (expected raw|zstd)",
                other
            )));
        }
        None => default_text_enc,
    };
    let dt = match dtype {
        Some("float32") => EmbeddingDType::Float32,
        Some("float16") => EmbeddingDType::Float16,
        Some("int8") => EmbeddingDType::Int8,
        Some(other) => {
            return Err(PyValueError::new_err(format!(
                "unknown dtype: {} (expected float32|float16|int8)",
                other
            )));
        }
        None => default_dtype,
    };
    let want_hnsw = with_hnsw.unwrap_or(default_hnsw);
    let want_bm25 = with_bm25.unwrap_or(default_bm25);

    let manifest = Manifest {
        embedding_model: embedding_model.to_string(),
        embedding_dim,
        n_chunks,
        chunker_version: chunker_version.to_string(),
        model_hash: model_hash.to_string(),
        title,
        version,
        created,
        description,
        authors,
        license,
        ..Default::default()
    };

    let mut builder = NestFileBuilder::new(manifest)
        .reproducible(reproducible)
        .with_provenance(provenance_value)
        .text_encoding(text_enc)
        .embedding_dtype(dt);

    // HNSW: build the index from f32 vectors (we have them in chunk_inputs
    // already). The runtime materializes f32 vectors at open time too —
    // here we use the originals so build is independent of dtype loss.
    if want_hnsw {
        let dim = embedding_dim as usize;
        let n = chunk_inputs.len();
        let mut flat: Vec<f32> = Vec::with_capacity(n * dim);
        for c in &chunk_inputs {
            flat.extend_from_slice(&c.embedding);
        }
        let idx = nest_runtime::ann::HnswIndex::build(
            flat,
            n,
            dim,
            hnsw_m,
            hnsw_ef_construction,
            hnsw_seed,
        );
        builder = builder.hnsw_index(idx.to_bytes());
    }

    if want_bm25 {
        let docs: Vec<String> = chunk_inputs
            .iter()
            .map(|c| c.canonical_text.clone())
            .collect();
        let bm = nest_runtime::bm25::Bm25Index::build(
            &docs,
            nest_runtime::bm25::DEFAULT_K1,
            nest_runtime::bm25::DEFAULT_B,
        );
        builder = builder.bm25_index(bm.to_bytes());
        // hybrid preset already set index_type via text path; otherwise
        // BM25 alone doesn't change the search path — runtime exposes
        // it explicitly via search_hybrid.
    }

    builder = builder.add_chunks(chunk_inputs);

    let bytes = builder
        .build_bytes()
        .map_err(|e| PyValueError::new_err(format!("{}", e)))?;
    std::fs::write(output_path, &bytes)
        .map_err(|e| PyValueError::new_err(format!("write {}: {}", output_path, e)))?;
    Ok(output_path.to_string())
}

/// Compute the canonical chunk_id for inputs that match the writer's
/// derivation. Useful for the Python-side ingestion to deduplicate chunks
/// before passing them in.
#[pyfunction]
fn chunk_id(
    canonical_text: &str,
    source_uri: &str,
    byte_start: u64,
    byte_end: u64,
    chunker_version: &str,
) -> String {
    nest_format::chunk_id(
        canonical_text,
        source_uri,
        byte_start,
        byte_end,
        chunker_version,
    )
}

#[pymodule]
fn _nest(m: &Bound<PyModule>) -> PyResult<()> {
    m.add_class::<NestFile>()?;
    m.add_class::<SearchHitPy>()?;
    m.add_function(wrap_pyfunction!(build, m)?)?;
    m.add_function(wrap_pyfunction!(chunk_id, m)?)?;
    Ok(())
}
