//! Builder for .nest v1 files.
//!
//! The builder owns structured chunk inputs and emits all six required
//! sections plus the manifest. Output is fully deterministic given the
//! same inputs.

use crate::chunk::{ChunkInput, chunk_id, validate_chunk};
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

/// High-level builder. Accepts canonical chunks plus an optional provenance
/// blob. Computes `chunk_id`s, lays out the file, writes deterministic bytes.
pub struct NestFileBuilder {
    manifest: Manifest,
    chunks: Vec<ChunkInput>,
    provenance: serde_json::Value,
    reproducible: bool,
}

impl NestFileBuilder {
    pub fn new(manifest: Manifest) -> Self {
        Self {
            manifest,
            chunks: Vec::new(),
            provenance: serde_json::json!({}),
            reproducible: false,
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

        let mut embeddings_bytes: Vec<u8> =
            Vec::with_capacity(self.chunks.len() * embedding_dim * 4);
        for c in &self.chunks {
            for v in &c.embedding {
                embeddings_bytes.extend_from_slice(&v.to_le_bytes());
            }
        }

        let contract = SearchContract {
            metric: self.manifest.metric.clone(),
            score_type: self.manifest.score_type.clone(),
            normalize: self.manifest.normalize.clone(),
            index_type: self.manifest.index_type.clone(),
            rerank_policy: self.manifest.rerank_policy.clone(),
        };

        // (id, name, payload) — names match `CANONICAL_SECTIONS` exactly.
        let mut sections: Vec<(u32, Vec<u8>)> = vec![
            (SECTION_CHUNK_IDS, encode_chunk_ids(&chunk_ids)?),
            (
                SECTION_CHUNKS_CANONICAL,
                encode_chunks_canonical(&canonical_texts)?,
            ),
            (
                SECTION_CHUNKS_ORIGINAL_SPANS,
                encode_chunks_original_spans(&original_spans)?,
            ),
            (SECTION_EMBEDDINGS, embeddings_bytes),
            (SECTION_PROVENANCE, encode_provenance(&self.provenance)?),
            (SECTION_SEARCH_CONTRACT, encode_search_contract(&contract)?),
        ];

        // Sanity: every required section is present and only those.
        debug_assert_eq!(sections.len(), REQUIRED_SECTIONS.len());

        // 4. Manifest JSON (canonical).
        let manifest_json = self.manifest.to_canonical_json()?;

        // 5. Layout offsets. Each section starts at a SECTION_ALIGNMENT
        //    boundary; the gap before is zero padding (already in the
        //    pre-zeroed buffer) and is NOT part of the section size.
        sections.sort_by_key(|s| s.0);
        let section_table_count = sections.len() as u64;
        let section_table_size = section_table_count * NEST_SECTION_ENTRY_SIZE as u64;
        let header_size = NEST_HEADER_SIZE as u64;
        let manifest_offset = header_size + section_table_size;
        let manifest_size = manifest_json.len() as u64;

        let mut section_entries = Vec::with_capacity(sections.len());
        let mut current_offset = align_up(manifest_offset + manifest_size, SECTION_ALIGNMENT);
        for (id, data) in &sections {
            section_entries.push(SectionEntry::new(*id, current_offset, data.len() as u64));
            let after = current_offset + data.len() as u64;
            current_offset = align_up(after, SECTION_ALIGNMENT);
        }

        // After the last section we want the footer immediately, so use
        // the unaligned end of the last section's data.
        let last_section_end = match (section_entries.last(), sections.last()) {
            (Some(entry), Some((_, data))) => entry.offset + data.len() as u64,
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
        //    is not hashed.
        for (i, (_, data)) in sections.iter().enumerate() {
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

        // And the manifest on disk must reflect the epoch override.
        let view = crate::reader::NestView::from_bytes(&b1).unwrap();
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
        let view = crate::reader::NestView::from_bytes(&bytes).unwrap();
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
}
