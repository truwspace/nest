//! Builder for .nest files.
//!
//! The builder owns structured chunk inputs and emits all six required
//! sections plus the manifest. Output is fully deterministic given the
//! same inputs and encoding choices.
//!
//! Encoding choices:
//!
//! - `SectionEncoding::Raw` (default) — sections stored verbatim.
//! - `SectionEncoding::Zstd` — text-heavy sections (canonical/spans/
//!   provenance/contract) are zstd-compressed on disk; the reader
//!   decompresses transparently.
//! - `EmbeddingDType::Float32 | Float16 | Int8` — controls the on-disk
//!   representation of the embeddings section. The runtime always
//!   accumulates dot products in f32 regardless of dtype.

use crate::chunk::{ChunkInput, chunk_id, validate_chunk};
use crate::encoding::{encode_int8_embeddings, f32_to_f16_bytes, zstd_encode};
use crate::error::NestError;
use crate::layout::*;
use crate::manifest::Manifest;
use crate::sections::{
    OriginalSpan, SearchContract, encode_chunk_ids, encode_chunks_canonical,
    encode_chunks_original_spans, encode_provenance, encode_search_contract,
};
use sha2::{Digest, Sha256};
use std::path::Path;

/// Timestamp written to the manifest when the builder is in reproducible
/// mode. Chosen so that two builds with identical inputs produce
/// identical bytes regardless of when they ran.
pub const REPRODUCIBLE_CREATED: &str = "1970-01-01T00:00:00Z";

/// Wire encoding choice for a non-embedding section. Embedding encoding
/// is controlled by `EmbeddingDType` since dtype and encoding are
/// intertwined.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SectionEncoding {
    Raw,
    Zstd,
}

impl SectionEncoding {
    pub fn id(self) -> u32 {
        match self {
            Self::Raw => SECTION_ENCODING_RAW,
            Self::Zstd => SECTION_ENCODING_ZSTD,
        }
    }
}

/// Embedding dtype + on-disk encoding. The two are 1:1 in v1 — float32
/// implies raw f32 LE, float16 implies raw f16 LE, int8 implies the
/// quantized prefix layout (see `encoding::encode_int8_embeddings`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum EmbeddingDType {
    Float32,
    Float16,
    Int8,
}

impl EmbeddingDType {
    pub fn manifest_str(self) -> &'static str {
        match self {
            Self::Float32 => "float32",
            Self::Float16 => "float16",
            Self::Int8 => "int8",
        }
    }
    pub fn encoding(self) -> u32 {
        match self {
            Self::Float32 => SECTION_ENCODING_RAW,
            Self::Float16 => SECTION_ENCODING_FLOAT16,
            Self::Int8 => SECTION_ENCODING_INT8,
        }
    }
}

/// High-level builder. Accepts canonical chunks plus an optional provenance
/// blob. Computes `chunk_id`s, lays out the file, writes deterministic bytes.
pub struct NestFileBuilder {
    manifest: Manifest,
    chunks: Vec<ChunkInput>,
    provenance: serde_json::Value,
    reproducible: bool,
    text_encoding: SectionEncoding,
    dtype: EmbeddingDType,
    /// Optional HNSW index payload, fully encoded by the caller. The
    /// builder doesn't know how to build an HNSW graph itself — that's
    /// the runtime's job.
    hnsw_index: Option<Vec<u8>>,
    bm25_index: Option<Vec<u8>>,
}

impl NestFileBuilder {
    pub fn new(manifest: Manifest) -> Self {
        Self {
            manifest,
            chunks: Vec::new(),
            provenance: serde_json::json!({}),
            reproducible: false,
            text_encoding: SectionEncoding::Raw,
            dtype: EmbeddingDType::Float32,
            hnsw_index: None,
            bm25_index: None,
        }
    }

    pub fn add_chunk(mut self, c: ChunkInput) -> Self {
        self.chunks.push(c);
        self
    }

    pub fn add_chunks<I: IntoIterator<Item = ChunkInput>>(mut self, chunks: I) -> Self {
        self.chunks.extend(chunks);
        self
    }

    pub fn with_provenance(mut self, v: serde_json::Value) -> Self {
        self.provenance = v;
        self
    }

    /// Reproducible build mode. When enabled, the writer overrides the
    /// manifest's `created` timestamp to `REPRODUCIBLE_CREATED` so that
    /// two builds with identical inputs produce byte-identical output.
    /// Provenance JSON is not rewritten — callers are responsible for
    /// keeping provenance deterministic if they want bit-for-bit equality.
    pub fn reproducible(mut self, on: bool) -> Self {
        self.reproducible = on;
        self
    }

    /// Encoding for text-heavy sections (chunks_canonical, original_spans,
    /// provenance, search_contract). `Zstd` shrinks PT-BR text by ~3-5×
    /// in practice. chunk_ids stays raw because it is high-entropy and
    /// almost incompressible.
    pub fn text_encoding(mut self, enc: SectionEncoding) -> Self {
        self.text_encoding = enc;
        self
    }

    /// Embedding dtype + on-disk encoding. Mutates the manifest's dtype
    /// to match. Quantized variants (`Float16`, `Int8`) are lossy; the
    /// runtime always accumulates dot products in f32.
    pub fn embedding_dtype(mut self, dt: EmbeddingDType) -> Self {
        self.dtype = dt;
        self.manifest.dtype = dt.manifest_str().to_string();
        self
    }

    /// Attach an HNSW index payload (already encoded by `nest-runtime`).
    /// Sets `index_type=hnsw`, `rerank_policy=exact`, `supports_ann=true`.
    pub fn hnsw_index(mut self, payload: Vec<u8>) -> Self {
        self.hnsw_index = Some(payload);
        self.manifest.index_type = "hnsw".into();
        self.manifest.rerank_policy = "exact".into();
        self.manifest.capabilities.supports_ann = true;
        self
    }

    /// Attach a BM25 index payload.
    pub fn bm25_index(mut self, payload: Vec<u8>) -> Self {
        self.bm25_index = Some(payload);
        self.manifest.capabilities.supports_bm25 = true;
        self
    }

    /// Mark the search path as hybrid (BM25 + cosine). Requires both an
    /// HNSW or exact path and a BM25 index. Caller is responsible for
    /// declaring `score_type=hybrid_rrf` if they want that, otherwise
    /// score_type stays "cosine".
    pub fn hybrid(mut self) -> Self {
        self.manifest.index_type = "hybrid".into();
        self.manifest.rerank_policy = "exact".into();
        self.manifest.score_type = "hybrid_rrf".into();
        self.manifest.capabilities.supports_bm25 = true;
        self
    }

    /// Build the file in memory. Pure computation — no I/O.
    pub fn build_bytes(mut self) -> crate::Result<Vec<u8>> {
        if self.reproducible {
            self.manifest.created = Some(REPRODUCIBLE_CREATED.into());
        }
        // 1. Validate manifest (rejects bad dtype/metric/etc up front).
        self.manifest.validate()?;

        // 2. Validate chunks against the manifest's embedding_dim and n_chunks.
        let embedding_dim = self.manifest.embedding_dim as usize;
        if self.chunks.len() as u64 != self.manifest.n_chunks {
            return Err(NestError::ManifestInvalid(format!(
                "n_chunks={} but builder has {} chunks",
                self.manifest.n_chunks,
                self.chunks.len()
            )));
        }
        for c in &self.chunks {
            validate_chunk(c, embedding_dim)?;
        }

        // 3. Derive section payloads.
        let chunk_ids: Vec<String> = self
            .chunks
            .iter()
            .map(|c| {
                chunk_id(
                    &c.canonical_text,
                    &c.source_uri,
                    c.byte_start,
                    c.byte_end,
                    &self.manifest.chunker_version,
                )
            })
            .collect();

        let canonical_texts: Vec<String> = self
            .chunks
            .iter()
            .map(|c| c.canonical_text.clone())
            .collect();
        let original_spans: Vec<OriginalSpan> = self
            .chunks
            .iter()
            .map(|c| OriginalSpan {
                source_uri: c.source_uri.clone(),
                byte_start: c.byte_start,
                byte_end: c.byte_end,
            })
            .collect();

        let n = self.chunks.len();
        let embeddings_bytes: Vec<u8> = match self.dtype {
            EmbeddingDType::Float32 => {
                let mut buf: Vec<u8> = Vec::with_capacity(n * embedding_dim * 4);
                for c in &self.chunks {
                    for v in &c.embedding {
                        buf.extend_from_slice(&v.to_le_bytes());
                    }
                }
                buf
            }
            EmbeddingDType::Float16 => {
                let mut flat: Vec<f32> = Vec::with_capacity(n * embedding_dim);
                for c in &self.chunks {
                    flat.extend_from_slice(&c.embedding);
                }
                f32_to_f16_bytes(&flat)
            }
            EmbeddingDType::Int8 => {
                let mut flat: Vec<f32> = Vec::with_capacity(n * embedding_dim);
                for c in &self.chunks {
                    flat.extend_from_slice(&c.embedding);
                }
                encode_int8_embeddings(&flat, n, embedding_dim)?
            }
        };

        let contract = SearchContract {
            metric: self.manifest.metric.clone(),
            score_type: self.manifest.score_type.clone(),
            normalize: self.manifest.normalize.clone(),
            index_type: self.manifest.index_type.clone(),
            rerank_policy: self.manifest.rerank_policy.clone(),
        };

        // (id, encoding, payload). chunk_ids stays raw (high-entropy SHA-256
        // hex strings, near-incompressible), embeddings get dtype-specific
        // encoding, everything else honors `text_encoding`.
        let text_enc = self.text_encoding;
        let mut sections: Vec<(u32, u32, Vec<u8>)> = Vec::with_capacity(8);
        sections.push((
            SECTION_CHUNK_IDS,
            SECTION_ENCODING_RAW,
            encode_chunk_ids(&chunk_ids)?,
        ));
        sections.push(maybe_zstd(
            SECTION_CHUNKS_CANONICAL,
            text_enc,
            encode_chunks_canonical(&canonical_texts)?,
        )?);
        sections.push(maybe_zstd(
            SECTION_CHUNKS_ORIGINAL_SPANS,
            text_enc,
            encode_chunks_original_spans(&original_spans)?,
        )?);
        sections.push((SECTION_EMBEDDINGS, self.dtype.encoding(), embeddings_bytes));
        sections.push(maybe_zstd(
            SECTION_PROVENANCE,
            text_enc,
            encode_provenance(&self.provenance)?,
        )?);
        sections.push(maybe_zstd(
            SECTION_SEARCH_CONTRACT,
            text_enc,
            encode_search_contract(&contract)?,
        )?);

        if let Some(payload) = self.hnsw_index.take() {
            // HNSW is binary, mostly random — zstd would barely help and
            // would defeat mmap-friendly reads. Always raw.
            sections.push((SECTION_HNSW_INDEX, SECTION_ENCODING_RAW, payload));
        }
        if let Some(payload) = self.bm25_index.take() {
            // BM25 posting lists are integer-heavy; zstd usually halves
            // them. Honor text_encoding here too.
            sections.push(maybe_zstd(SECTION_BM25_INDEX, text_enc, payload)?);
        }

        // Sanity: every required section is present (writer never drops one).
        debug_assert!(
            REQUIRED_SECTIONS
                .iter()
                .all(|(id, _)| sections.iter().any(|s| s.0 == *id))
        );

        // 4. Manifest JSON (canonical).
        let manifest_json = self.manifest.to_canonical_json()?;

        // 5. Layout offsets. Each section starts at SECTION_ALIGNMENT.
        sections.sort_by_key(|s| s.0);
        let section_table_count = sections.len() as u64;
        let section_table_size = section_table_count * NEST_SECTION_ENTRY_SIZE as u64;
        let header_size = NEST_HEADER_SIZE as u64;
        let manifest_offset = header_size + section_table_size;
        let manifest_size = manifest_json.len() as u64;

        let mut section_entries: Vec<SectionEntry> = Vec::with_capacity(sections.len());
        let mut current_offset = align_up(manifest_offset + manifest_size, SECTION_ALIGNMENT);
        for (id, encoding, data) in &sections {
            let mut entry = SectionEntry::new(*id, current_offset, data.len() as u64);
            entry.encoding = *encoding;
            section_entries.push(entry);
            let after = current_offset + data.len() as u64;
            current_offset = align_up(after, SECTION_ALIGNMENT);
        }

        // After the last section we want the footer immediately, so use
        // the unaligned end of the last section's data.
        let last_section_end = match (section_entries.last(), sections.last()) {
            (Some(entry), Some((_, _, data))) => entry.offset + data.len() as u64,
            _ => manifest_offset + manifest_size,
        };
        let data_end = last_section_end as usize;
        let file_size = data_end + NEST_FOOTER_SIZE;
        let mut buf = vec![0u8; file_size];

        // 6. Write header (placeholder; checksum recomputed at end).
        let mut header = NestHeader::new(
            self.manifest.embedding_dim,
            self.manifest.n_chunks,
            self.chunks.len() as u64,
            file_size as u64,
            header_size,
            section_table_count,
            manifest_offset,
            manifest_size,
        );
        buf[..NEST_HEADER_SIZE].copy_from_slice(header.as_bytes());

        // 7. Write section table (placeholder checksums; filled below).
        for (i, entry) in section_entries.iter().enumerate() {
            let off = NEST_HEADER_SIZE + i * NEST_SECTION_ENTRY_SIZE;
            buf[off..off + NEST_SECTION_ENTRY_SIZE].copy_from_slice(entry.as_bytes());
        }

        // 8. Write manifest.
        let manifest_off = manifest_offset as usize;
        buf[manifest_off..manifest_off + manifest_json.len()].copy_from_slice(&manifest_json);

        // 9. Write section data at its declared (aligned) offset and
        //    compute checksums over data only — padding stays zero and
        //    is not hashed. Section checksum hashes the **physical** bytes
        //    on disk (so for zstd sections it's over the compressed bytes).
        for (i, (_, _, data)) in sections.iter().enumerate() {
            let entry_off = NEST_HEADER_SIZE + i * NEST_SECTION_ENTRY_SIZE;
            let data_off = section_entries[i].offset as usize;
            buf[data_off..data_off + data.len()].copy_from_slice(data);
            let hash = Sha256::digest(data);
            buf[entry_off + 24..entry_off + 32].copy_from_slice(&hash[..8]);
        }

        // 10. Footer hash (covers everything before footer).
        let footer_hash = NestFooter::compute_file_hash(&buf[..data_end]);
        buf[data_end..data_end + 8].copy_from_slice(&NEST_FOOTER_SIZE.to_le_bytes());
        buf[data_end + 8..file_size].copy_from_slice(&footer_hash);

        // 11. Recompute header checksum (file_size already final).
        header.compute_checksum();
        buf[..NEST_HEADER_SIZE].copy_from_slice(header.as_bytes());

        Ok(buf)
    }

    pub fn write_to_path(self, path: impl AsRef<Path>) -> crate::Result<()> {
        let buf = self.build_bytes()?;
        std::fs::write(path, buf)?;
        Ok(())
    }
}

fn maybe_zstd(
    section_id: u32,
    enc: SectionEncoding,
    payload: Vec<u8>,
) -> crate::Result<(u32, u32, Vec<u8>)> {
    Ok(match enc {
        SectionEncoding::Raw => (section_id, SECTION_ENCODING_RAW, payload),
        SectionEncoding::Zstd => (section_id, SECTION_ENCODING_ZSTD, zstd_encode(&payload)?),
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::reader::NestView;

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

    fn one_chunk() -> ChunkInput {
        ChunkInput {
            canonical_text: "hello world".into(),
            source_uri: "doc.txt".into(),
            byte_start: 0,
            byte_end: 11,
            embedding: vec![1.0, 0.0, 0.0, 0.0],
        }
    }

    #[test]
    fn deterministic_build() {
        let b1 = NestFileBuilder::new(good_manifest())
            .add_chunk(one_chunk())
            .build_bytes()
            .unwrap();
        let b2 = NestFileBuilder::new(good_manifest())
            .add_chunk(one_chunk())
            .build_bytes()
            .unwrap();
        assert_eq!(b1, b2);
    }

    #[test]
    fn reproducible_mode_overrides_created() {
        let mut m1 = good_manifest();
        m1.created = Some("2026-04-27T13:30:00Z".into());
        let mut m2 = good_manifest();
        m2.created = Some("2030-12-31T23:59:59Z".into());

        let b1 = NestFileBuilder::new(m1)
            .reproducible(true)
            .add_chunk(one_chunk())
            .build_bytes()
            .unwrap();
        let b2 = NestFileBuilder::new(m2)
            .reproducible(true)
            .add_chunk(one_chunk())
            .build_bytes()
            .unwrap();
        assert_eq!(b1, b2, "reproducible builds must be byte-equal");

        let view = NestView::from_bytes(&b1).unwrap();
        assert_eq!(view.manifest.created.as_deref(), Some(REPRODUCIBLE_CREATED));
    }

    #[test]
    fn non_reproducible_preserves_created() {
        let mut m = good_manifest();
        m.created = Some("2026-04-27T13:30:00Z".into());
        let bytes = NestFileBuilder::new(m.clone())
            .add_chunk(one_chunk())
            .build_bytes()
            .unwrap();
        let view = NestView::from_bytes(&bytes).unwrap();
        assert_eq!(
            view.manifest.created.as_deref(),
            Some("2026-04-27T13:30:00Z")
        );
    }

    #[test]
    fn rejects_chunk_count_mismatch() {
        let mut m = good_manifest();
        m.n_chunks = 2;
        let err = NestFileBuilder::new(m).add_chunk(one_chunk()).build_bytes();
        assert!(matches!(err, Err(NestError::ManifestInvalid(_))));
    }

    #[test]
    fn rejects_bad_dim() {
        let mut c = one_chunk();
        c.embedding = vec![1.0, 0.0]; // wrong dim
        let err = NestFileBuilder::new(good_manifest())
            .add_chunk(c)
            .build_bytes();
        assert!(matches!(err, Err(NestError::DimensionMismatch { .. })));
    }

    #[test]
    fn zstd_encoding_preserves_content_hash() {
        // The whole point of decoded-content hashing: same logical
        // content should yield the same content_hash whether we zstd
        // it or not. file_hash will differ (different bytes on disk).
        let mut m = good_manifest();
        m.n_chunks = 3;
        let chunks: Vec<ChunkInput> = (0..3)
            .map(|i| ChunkInput {
                canonical_text: format!("paragraph number {}", i),
                source_uri: "doc.txt".into(),
                byte_start: i * 10,
                byte_end: i * 10 + 5,
                embedding: {
                    let mut v = vec![0.0; 4];
                    v[(i % 4) as usize] = 1.0;
                    v
                },
            })
            .collect();

        let raw = NestFileBuilder::new(m.clone())
            .add_chunks(chunks.clone())
            .build_bytes()
            .unwrap();
        let zst = NestFileBuilder::new(m)
            .text_encoding(SectionEncoding::Zstd)
            .add_chunks(chunks)
            .build_bytes()
            .unwrap();

        let v_raw = NestView::from_bytes(&raw).unwrap();
        let v_zst = NestView::from_bytes(&zst).unwrap();
        assert_eq!(
            v_raw.content_hash_hex().unwrap(),
            v_zst.content_hash_hex().unwrap(),
            "content_hash must equal across wire encodings"
        );
        assert_ne!(
            v_raw.file_hash_hex(),
            v_zst.file_hash_hex(),
            "file_hash must differ between raw and zstd"
        );
    }

    #[test]
    fn float16_embeddings_roundtrip() {
        let mut m = good_manifest();
        m.n_chunks = 2;
        let bytes = NestFileBuilder::new(m)
            .embedding_dtype(EmbeddingDType::Float16)
            .add_chunk(ChunkInput {
                canonical_text: "a".into(),
                source_uri: "d".into(),
                byte_start: 0,
                byte_end: 1,
                embedding: vec![1.0, 0.0, 0.0, 0.0],
            })
            .add_chunk(ChunkInput {
                canonical_text: "b".into(),
                source_uri: "d".into(),
                byte_start: 1,
                byte_end: 2,
                embedding: vec![0.0, 1.0, 0.0, 0.0],
            })
            .build_bytes()
            .unwrap();
        let view = NestView::from_bytes(&bytes).unwrap();
        assert_eq!(view.manifest.dtype, "float16");
        let entry = view.entry(SECTION_EMBEDDINGS).unwrap();
        assert_eq!(entry.encoding, SECTION_ENCODING_FLOAT16);
        assert_eq!(entry.size as usize, 2 * 4 * 2); // n*dim*2
    }

    #[test]
    fn int8_embeddings_roundtrip() {
        let mut m = good_manifest();
        m.n_chunks = 2;
        let bytes = NestFileBuilder::new(m)
            .embedding_dtype(EmbeddingDType::Int8)
            .add_chunk(ChunkInput {
                canonical_text: "a".into(),
                source_uri: "d".into(),
                byte_start: 0,
                byte_end: 1,
                embedding: vec![1.0, 0.0, 0.0, 0.0],
            })
            .add_chunk(ChunkInput {
                canonical_text: "b".into(),
                source_uri: "d".into(),
                byte_start: 1,
                byte_end: 2,
                embedding: vec![0.0, 1.0, 0.0, 0.0],
            })
            .build_bytes()
            .unwrap();
        let view = NestView::from_bytes(&bytes).unwrap();
        assert_eq!(view.manifest.dtype, "int8");
        let entry = view.entry(SECTION_EMBEDDINGS).unwrap();
        assert_eq!(entry.encoding, SECTION_ENCODING_INT8);
        // 8-byte prefix + n*4 (scales) + n*dim
        assert_eq!(entry.size as usize, 8 + 2 * 4 + 2 * 4);
    }

    #[test]
    fn rejects_zstd_embeddings() {
        // Embeddings can never be zstd-compressed (we want SIMD-friendly
        // mmap reads). text_encoding does not apply to embeddings.
        let m = good_manifest();
        let bytes = NestFileBuilder::new(m)
            .text_encoding(SectionEncoding::Zstd)
            .add_chunk(one_chunk())
            .build_bytes()
            .unwrap();
        let view = NestView::from_bytes(&bytes).unwrap();
        let entry = view.entry(SECTION_EMBEDDINGS).unwrap();
        assert_eq!(entry.encoding, SECTION_ENCODING_RAW);
    }
}
