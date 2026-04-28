//! mmap-backed runtime for `.nest` files.
//!
//! Owns the mmap so `MmapNestFile` is `'static`. Section metadata is parsed
//! once at open time using `NestView`, then the mmap is moved into `Self`
//! and embeddings are read directly from `&self._mmap[offset..]`.
//!
//! Supports float32 / float16 / int8 dtypes with a SIMD dispatcher
//! (AVX2 / NEON / scalar). Optional ANN (`hnsw`) and lexical (`bm25`)
//! sections rerank into the exact cosine path so the final score is
//! always the real cosine value.

use memmap2::Mmap;
use nest_format::Int8EmbeddingsView;
use nest_format::layout::*;
use nest_format::reader::NestView;
use nest_format::sections::{OriginalSpan, decode_chunk_ids, decode_chunks_original_spans};
use std::path::Path;

pub mod ann;
pub mod bm25;
pub mod error;
pub mod simd;

pub use error::RuntimeError;
pub use simd::SimdBackend;

#[derive(Clone, Debug, PartialEq)]
pub struct SearchHit {
    pub chunk_id: String,
    pub score: f32,
    pub score_type: &'static str,
    pub source_uri: String,
    pub offset_start: u64,
    pub offset_end: u64,
    pub embedding_model: String,
    pub index_type: &'static str,
    pub reranked: bool,
    pub file_hash: String,
    pub content_hash: String,
    pub citation_id: String,
}

#[derive(Clone, Debug, PartialEq)]
pub struct SearchResult {
    pub hits: Vec<SearchHit>,
    pub query_time_ms: f64,
    pub index_type: &'static str,
    pub recall: f32,
    pub truncated: bool,
    pub k_requested: i32,
    pub k_returned: usize,
}

/// Runtime view of the embeddings section dtype.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DType {
    Float32,
    Float16,
    Int8,
}

impl DType {
    fn from_str(s: &str) -> Result<Self, RuntimeError> {
        match s {
            "float32" => Ok(Self::Float32),
            "float16" => Ok(Self::Float16),
            "int8" => Ok(Self::Int8),
            other => Err(RuntimeError::Format(
                nest_format::NestError::UnsupportedDType(other.into()),
            )),
        }
    }
    pub fn bytes_per_value(self) -> usize {
        match self {
            Self::Float32 => 4,
            Self::Float16 => 2,
            Self::Int8 => 1,
        }
    }
    pub fn name(self) -> &'static str {
        match self {
            Self::Float32 => "float32",
            Self::Float16 => "float16",
            Self::Int8 => "int8",
        }
    }
}

pub struct MmapNestFile {
    _mmap: Mmap,
    embedding_dim: usize,
    n_embeddings: usize,
    dtype: DType,
    /// Byte offset (within the mmap) of the embeddings section payload.
    embeddings_offset: usize,
    /// Total physical bytes of the embeddings section.
    embeddings_size: usize,
    chunk_ids: Vec<String>,
    spans: Vec<OriginalSpan>,
    embedding_model: String,
    file_hash: String,
    content_hash: String,
    /// Optional ANN index. Built from the HNSW section payload at open
    /// time (eager: build cost is paid once, queries get fast path).
    ann_index: Option<ann::HnswIndex>,
    /// Optional BM25 index. Mostly tiny ints; deserialized eagerly.
    bm25_index: Option<bm25::Bm25Index>,
    /// What the manifest says the search path is. The runtime honors
    /// this at search time.
    declared_index_type: String,
    declared_score_type: String,
}

impl MmapNestFile {
    pub fn open(path: &Path) -> Result<Self, RuntimeError> {
        let file = std::fs::File::open(path)?;
        let mmap = unsafe { Mmap::map(&file)? };
        let view = NestView::from_bytes(&mmap)?;
        view.validate_embeddings_values()?;

        let dim = view.header.embedding_dim as usize;
        let n = view.header.n_embeddings as usize;
        let dtype = DType::from_str(&view.manifest.dtype)?;

        let entry = view.entry(SECTION_EMBEDDINGS)?;
        let embeddings_offset = entry.offset as usize;
        let embeddings_size = entry.size as usize;

        // Decoded chunk_ids / spans (handles zstd transparently).
        let chunk_ids = decode_chunk_ids(&view.decoded_section(SECTION_CHUNK_IDS)?, n)?;
        let spans =
            decode_chunks_original_spans(&view.decoded_section(SECTION_CHUNKS_ORIGINAL_SPANS)?, n)?;

        // Optional ANN section. Materialize f32 vectors from the
        // embeddings section so the graph can compute distances at
        // search time independent of the on-disk dtype.
        let ann_index = if view
            .section_table
            .iter()
            .any(|e| e.section_id == SECTION_HNSW_INDEX)
        {
            let bytes = view.decoded_section(SECTION_HNSW_INDEX)?;
            let mut idx = ann::HnswIndex::from_bytes(&bytes, n, dim)?;
            let emb_bytes = view.get_section_data(SECTION_EMBEDDINGS)?;
            let vectors = materialize_f32_vectors(&view.manifest.dtype, emb_bytes, n, dim)?;
            idx.attach_vectors(vectors);
            Some(idx)
        } else {
            None
        };

        let bm25_index = if view
            .section_table
            .iter()
            .any(|e| e.section_id == SECTION_BM25_INDEX)
        {
            let bytes = view.decoded_section(SECTION_BM25_INDEX)?;
            Some(bm25::Bm25Index::from_bytes(&bytes)?)
        } else {
            None
        };

        let embedding_model = view.manifest.embedding_model.clone();
        let declared_index_type = view.manifest.index_type.clone();
        let declared_score_type = view.manifest.score_type.clone();
        let file_hash = view.file_hash_hex();
        let content_hash = view.content_hash_hex()?;
        drop(view);

        Ok(Self {
            _mmap: mmap,
            embedding_dim: dim,
            n_embeddings: n,
            dtype,
            embeddings_offset,
            embeddings_size,
            chunk_ids,
            spans,
            embedding_model,
            file_hash,
            content_hash,
            ann_index,
            bm25_index,
            declared_index_type,
            declared_score_type,
        })
    }

    pub fn embedding_dim(&self) -> usize {
        self.embedding_dim
    }
    pub fn n_embeddings(&self) -> usize {
        self.n_embeddings
    }
    pub fn dtype(&self) -> DType {
        self.dtype
    }
    pub fn file_hash(&self) -> &str {
        &self.file_hash
    }
    pub fn content_hash(&self) -> &str {
        &self.content_hash
    }
    pub fn simd_backend(&self) -> SimdBackend {
        simd::detect_backend()
    }
    pub fn declared_index_type(&self) -> &str {
        &self.declared_index_type
    }
    pub fn declared_score_type(&self) -> &str {
        &self.declared_score_type
    }
    pub fn has_ann(&self) -> bool {
        self.ann_index.is_some()
    }
    pub fn has_bm25(&self) -> bool {
        self.bm25_index.is_some()
    }

    /// Re-parse the mmap and return a JSON document mirroring `nest
    /// inspect`: header fields, section table entries, manifest, hashes,
    /// and the runtime SIMD backend.
    pub fn inspect_json(&self) -> Result<String, RuntimeError> {
        let view = NestView::from_bytes(&self._mmap)?;
        let magic = std::str::from_utf8(&view.header.magic)
            .unwrap_or("")
            .to_string();
        let sections: Vec<serde_json::Value> = view
            .section_table
            .iter()
            .map(|e| {
                let name = nest_format::layout::section_name(e.section_id).unwrap_or("unknown");
                serde_json::json!({
                    "section_id": e.section_id,
                    "name": name,
                    "encoding": e.encoding,
                    "offset": e.offset,
                    "size": e.size,
                    "checksum": hex::encode(e.checksum),
                })
            })
            .collect();
        let doc = serde_json::json!({
            "magic": magic,
            "version_major": view.header.version_major,
            "version_minor": view.header.version_minor,
            "format_version": view.manifest.format_version,
            "schema_version": view.manifest.schema_version,
            "embedding_dim": view.header.embedding_dim,
            "n_chunks": view.header.n_chunks,
            "n_embeddings": view.header.n_embeddings,
            "file_size": view.header.file_size,
            "manifest": view.manifest,
            "sections": sections,
            "file_hash": view.file_hash_hex(),
            "content_hash": view.content_hash_hex()?,
            "simd_backend": self.simd_backend().name(),
        });
        serde_json::to_string(&doc)
            .map_err(|e| RuntimeError::Format(nest_format::NestError::Json(e)))
    }

    /// Re-run all reader-side validation. The file was already
    /// validated at `open()` time, but callers can invoke this
    /// explicitly to detect tampering after the fact (e.g. the mmap
    /// pages got swapped under the runtime).
    pub fn revalidate(&self) -> Result<(), RuntimeError> {
        let view = NestView::from_bytes(&self._mmap)?;
        view.validate_embeddings_values()?;
        let _ = view.search_contract()?;
        Ok(())
    }

    fn embeddings_bytes(&self) -> &[u8] {
        &self._mmap[self.embeddings_offset..self.embeddings_offset + self.embeddings_size]
    }

    fn validate_query(&self, query: &[f32], k: i32) -> Result<Vec<f32>, RuntimeError> {
        if k <= 0 {
            return Err(RuntimeError::InvalidK(k));
        }
        if query.is_empty() {
            return Err(RuntimeError::EmptyQuery);
        }
        if query.len() != self.embedding_dim {
            return Err(RuntimeError::DimensionMismatch {
                expected: self.embedding_dim,
                got: query.len(),
            });
        }
        for &v in query {
            if v.is_nan() || v.is_infinite() {
                return Err(RuntimeError::InvalidQueryValue);
            }
        }
        let mut qnorm = query.to_vec();
        let norm: f32 = qnorm.iter().map(|x| x * x).sum::<f32>().sqrt();
        if norm == 0.0 {
            return Err(RuntimeError::ZeroNormQuery);
        }
        for x in &mut qnorm {
            *x /= norm;
        }
        Ok(qnorm)
    }

    /// Score every chunk against `qnorm` using the dtype-specific dot
    /// product. Returns `(idx, score)` pairs in the natural index order.
    fn score_all(&self, qnorm: &[f32]) -> Result<Vec<(usize, f32)>, RuntimeError> {
        let n = self.n_embeddings;
        let dim = self.embedding_dim;
        let bytes = self.embeddings_bytes();
        let mut scores: Vec<(usize, f32)> = Vec::with_capacity(n);
        match self.dtype {
            DType::Float32 => {
                let row_size = dim * 4;
                for i in 0..n {
                    let off = i * row_size;
                    let s = simd::dot_f32_bytes(qnorm, &bytes[off..off + row_size]);
                    scores.push((i, s));
                }
            }
            DType::Float16 => {
                let row_size = dim * 2;
                for i in 0..n {
                    let off = i * row_size;
                    let s = simd::dot_f32_f16_bytes(qnorm, &bytes[off..off + row_size]);
                    scores.push((i, s));
                }
            }
            DType::Int8 => {
                let view =
                    Int8EmbeddingsView::parse(bytes, n, dim).map_err(RuntimeError::Format)?;
                for i in 0..n {
                    let scale = view.scale(i);
                    let row = view.row(i);
                    let s = simd::dot_f32_i8(qnorm, row, scale);
                    scores.push((i, s));
                }
            }
        }
        Ok(scores)
    }

    /// Score a sliced subset of indices (used by ANN/BM25 rerank). The
    /// returned vector mirrors `idxs.len()` in order.
    fn score_subset(
        &self,
        qnorm: &[f32],
        idxs: &[usize],
    ) -> Result<Vec<(usize, f32)>, RuntimeError> {
        let dim = self.embedding_dim;
        let bytes = self.embeddings_bytes();
        let mut out: Vec<(usize, f32)> = Vec::with_capacity(idxs.len());
        match self.dtype {
            DType::Float32 => {
                let row_size = dim * 4;
                for &i in idxs {
                    let off = i * row_size;
                    out.push((i, simd::dot_f32_bytes(qnorm, &bytes[off..off + row_size])));
                }
            }
            DType::Float16 => {
                let row_size = dim * 2;
                for &i in idxs {
                    let off = i * row_size;
                    out.push((
                        i,
                        simd::dot_f32_f16_bytes(qnorm, &bytes[off..off + row_size]),
                    ));
                }
            }
            DType::Int8 => {
                let view = Int8EmbeddingsView::parse(bytes, self.n_embeddings, dim)
                    .map_err(RuntimeError::Format)?;
                for &i in idxs {
                    let scale = view.scale(i);
                    let row = view.row(i);
                    out.push((i, simd::dot_f32_i8(qnorm, row, scale)));
                }
            }
        }
        Ok(out)
    }

    /// Exact flat search. The recall=1.0 ground truth.
    pub fn search(&self, query: &[f32], k: i32) -> Result<SearchResult, RuntimeError> {
        let t0 = std::time::Instant::now();
        let qnorm = self.validate_query(query, k)?;
        let mut scores = self.score_all(&qnorm)?;
        sort_scores_desc(&mut scores);
        let k_usize = k as usize;
        let truncated = k_usize < self.n_embeddings;
        let top = &scores[..k_usize.min(self.n_embeddings)];
        let hits = self.materialize_hits(top, "exact", false);
        Ok(SearchResult {
            hits: hits.clone(),
            query_time_ms: t0.elapsed().as_secs_f64() * 1000.0,
            index_type: "exact",
            recall: 1.0,
            truncated,
            k_requested: k,
            k_returned: hits.len(),
        })
    }

    /// ANN search. Pulls `ef_search` candidates from HNSW, reranks with
    /// the exact dot product, returns top-k. Falls back to `search()` if
    /// no ANN section is present.
    pub fn search_ann(
        &self,
        query: &[f32],
        k: i32,
        ef_search: usize,
    ) -> Result<SearchResult, RuntimeError> {
        let t0 = std::time::Instant::now();
        let Some(idx) = self.ann_index.as_ref() else {
            return self.search(query, k);
        };
        let qnorm = self.validate_query(query, k)?;
        let candidates = idx.search(&qnorm, ef_search.max(k as usize));
        let mut reranked = self.score_subset(&qnorm, &candidates)?;
        sort_scores_desc(&mut reranked);
        let k_usize = k as usize;
        let truncated = k_usize < self.n_embeddings;
        let top = &reranked[..k_usize.min(reranked.len())];
        let hits = self.materialize_hits(top, "hnsw", true);
        Ok(SearchResult {
            hits: hits.clone(),
            query_time_ms: t0.elapsed().as_secs_f64() * 1000.0,
            index_type: "hnsw",
            // Recall is candidate-set dependent; runtime caller can
            // measure it against `search()` directly. We don't lie here.
            recall: f32::NAN,
            truncated,
            k_requested: k,
            k_returned: hits.len(),
        })
    }

    /// Hybrid search: BM25 candidates ∪ ANN (or exact) candidates,
    /// reciprocal-rank fusion, then exact cosine rerank on the union.
    /// Final score is the real cosine.
    pub fn search_hybrid(
        &self,
        query_vec: &[f32],
        query_text: &str,
        k: i32,
        candidates_per_path: usize,
    ) -> Result<SearchResult, RuntimeError> {
        let t0 = std::time::Instant::now();
        let qnorm = self.validate_query(query_vec, k)?;

        // Vector path: ANN if available, otherwise top-`candidates`.
        let vec_candidates: Vec<usize> = if let Some(idx) = self.ann_index.as_ref() {
            idx.search(&qnorm, candidates_per_path.max(k as usize))
        } else {
            let mut all = self.score_all(&qnorm)?;
            sort_scores_desc(&mut all);
            all.iter().take(candidates_per_path).map(|p| p.0).collect()
        };

        // Lexical path.
        let lex_candidates: Vec<usize> = if let Some(bm) = self.bm25_index.as_ref() {
            bm.search(query_text, candidates_per_path)
                .into_iter()
                .map(|(idx, _score)| idx)
                .collect()
        } else {
            Vec::new()
        };

        // Reciprocal-rank fusion to pick a union, then exact rerank.
        let union = bm25::rrf_union(&vec_candidates, &lex_candidates);
        let mut reranked = self.score_subset(&qnorm, &union)?;
        sort_scores_desc(&mut reranked);
        let k_usize = k as usize;
        let truncated = k_usize < self.n_embeddings;
        let top = &reranked[..k_usize.min(reranked.len())];
        let hits = self.materialize_hits(top, "hybrid", true);
        Ok(SearchResult {
            hits: hits.clone(),
            query_time_ms: t0.elapsed().as_secs_f64() * 1000.0,
            index_type: "hybrid",
            recall: f32::NAN,
            truncated,
            k_requested: k,
            k_returned: hits.len(),
        })
    }

    fn materialize_hits(
        &self,
        scored: &[(usize, f32)],
        index_type: &'static str,
        reranked: bool,
    ) -> Vec<SearchHit> {
        scored
            .iter()
            .map(|(idx, score)| {
                let span = &self.spans[*idx];
                let id = &self.chunk_ids[*idx];
                SearchHit {
                    chunk_id: id.clone(),
                    score: *score,
                    score_type: "cosine",
                    source_uri: span.source_uri.clone(),
                    offset_start: span.byte_start,
                    offset_end: span.byte_end,
                    embedding_model: self.embedding_model.clone(),
                    index_type,
                    reranked,
                    file_hash: self.file_hash.clone(),
                    content_hash: self.content_hash.clone(),
                    citation_id: format!("nest://{}/{}", self.content_hash, id),
                }
            })
            .collect()
    }
}

#[inline]
fn sort_scores_desc(scores: &mut [(usize, f32)]) {
    scores.sort_by(|a, b| {
        b.1.partial_cmp(&a.1)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then_with(|| a.0.cmp(&b.0))
    });
}

/// Decode the embeddings section into a flat row-major f32 buffer
/// regardless of the on-disk dtype. Used by the ANN index since the
/// HNSW graph accumulates distances in f32. Cost: one allocation of
/// `n * dim * 4` bytes — the price of being dtype-agnostic.
fn materialize_f32_vectors(
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
        other => Err(RuntimeError::Format(
            nest_format::NestError::UnsupportedDType(other.into()),
        )),
    }
}
