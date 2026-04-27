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

    #[getter]
    fn embedding_dim(&self) -> usize {
        self.rt.embedding_dim()
    }

    #[getter]
    fn n_embeddings(&self) -> usize {
        self.rt.n_embeddings()
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

/// Build a .nest v1 file from already-embedded chunks.
///
/// `chunks` is a list of dicts with keys:
///   - canonical_text: str
///   - source_uri: str
///   - byte_start: int
///   - byte_end: int
///   - embedding: list[float] (length == embedding_dim)
///
/// Optional manifest fields (`title`, `version`, `created`, `description`,
/// `authors`, `license`) and a free-form `provenance` dict can be passed
/// as kwargs. `reproducible=True` overrides `created` so two builds with
/// identical inputs produce byte-identical output.
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
) -> PyResult<String> {
    use nest_format::ChunkInput;
    use nest_format::manifest::Manifest;
    use nest_format::writer::NestFileBuilder;

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

    let bytes = NestFileBuilder::new(manifest)
        .reproducible(reproducible)
        .with_provenance(provenance_value)
        .add_chunks(chunk_inputs)
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
