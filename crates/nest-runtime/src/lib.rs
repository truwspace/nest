//! mmap-backed runtime for `.nest` v1 files.
//!
//! Owns the mmap so `MmapNestFile` is `'static`. Section metadata is parsed
//! once at open time using `NestView`, then the mmap is moved into `Self`
//! and embeddings are read directly from `&self._mmap[offset..]`.

use memmap2::Mmap;
use nest_format::layout::*;
use nest_format::reader::NestView;
use nest_format::sections::{OriginalSpan, decode_chunk_ids, decode_chunks_original_spans};
use std::path::Path;

pub mod error;
pub use error::RuntimeError;

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

pub struct MmapNestFile {
    _mmap: Mmap,
    embedding_dim: usize,
    n_embeddings: usize,
    embeddings_offset: usize,
    chunk_ids: Vec<String>,
    spans: Vec<OriginalSpan>,
    embedding_model: String,
    file_hash: String,
    content_hash: String,
}

impl MmapNestFile {
    pub fn open(path: &Path) -> Result<Self, RuntimeError> {
        let file = std::fs::File::open(path)?;
        let mmap = unsafe { Mmap::map(&file)? };
        let view = NestView::from_bytes(&mmap)?;
        view.validate_embeddings_values()?;

        let dim = view.header.embedding_dim as usize;
        let n = view.header.n_embeddings as usize;

        let emb = view.get_section_data(SECTION_EMBEDDINGS)?;
        let embeddings_offset = emb.as_ptr() as usize - mmap.as_ptr() as usize;

        // No silent fallback: missing/malformed required sections propagate.
        let chunk_ids = decode_chunk_ids(view.get_section_data(SECTION_CHUNK_IDS)?, n)?;
        let spans =
            decode_chunks_original_spans(view.get_section_data(SECTION_CHUNKS_ORIGINAL_SPANS)?, n)?;

        let embedding_model = view.manifest.embedding_model.clone();
        let file_hash = view.file_hash_hex();
        let content_hash = view.content_hash_hex();
        drop(view);

        Ok(Self {
            _mmap: mmap,
            embedding_dim: dim,
            n_embeddings: n,
            embeddings_offset,
            chunk_ids,
            spans,
            embedding_model,
            file_hash,
            content_hash,
        })
    }

    pub fn embedding_dim(&self) -> usize {
        self.embedding_dim
    }
    pub fn n_embeddings(&self) -> usize {
        self.n_embeddings
    }
    pub fn file_hash(&self) -> &str {
        &self.file_hash
    }
    pub fn content_hash(&self) -> &str {
        &self.content_hash
    }

    pub fn search(&self, query: &[f32], k: i32) -> Result<SearchResult, RuntimeError> {
        let t0 = std::time::Instant::now();
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

        let k = k as usize;
        let dim = self.embedding_dim;
        let n = self.n_embeddings;
        let data = self._mmap.as_ref();
        let base = self.embeddings_offset;

        // Flat exact search. Sort+truncate is acceptable as the v1 baseline;
        // a top-k heap can replace this once profiling shows it matters.
        let mut scores: Vec<(usize, f32)> = Vec::with_capacity(n);
        for i in 0..n {
            let off = base + i * dim * 4;
            let mut dot = 0.0f32;
            for j in 0..dim {
                let val = f32::from_le_bytes([
                    data[off + j * 4],
                    data[off + j * 4 + 1],
                    data[off + j * 4 + 2],
                    data[off + j * 4 + 3],
                ]);
                dot += qnorm[j] * val;
            }
            scores.push((i, dot));
        }

        scores.sort_by(|a, b| {
            b.1.partial_cmp(&a.1)
                .unwrap_or(std::cmp::Ordering::Equal)
                .then_with(|| a.0.cmp(&b.0))
        });

        let truncated = k < n;
        let top = &scores[..k.min(n)];

        let hits: Vec<SearchHit> = top
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
                    index_type: "exact",
                    reranked: false,
                    file_hash: self.file_hash.clone(),
                    content_hash: self.content_hash.clone(),
                    citation_id: format!("nest://{}/{}", self.content_hash, id),
                }
            })
            .collect();

        let k_returned = hits.len();
        Ok(SearchResult {
            hits,
            query_time_ms: t0.elapsed().as_secs_f64() * 1000.0,
            index_type: "exact",
            recall: 1.0,
            truncated,
            k_requested: k as i32,
            k_returned,
        })
    }
}
