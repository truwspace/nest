//! Zero-copy view over a `.nest` byte slice.
//!
//! The reader does no I/O — callers pass an `&[u8]` (e.g. backed by an
//! `mmap`). Parsing validates magic, header checksum, file_size, all
//! section checksums, footer hash, manifest schema, and the presence of
//! every required section.

use crate::error::NestError;
use crate::layout::*;
use crate::manifest::Manifest;
use crate::sections::{SearchContract, decode_search_contract};

pub struct NestView<'a> {
    data: &'a [u8],
    pub header: NestHeader,
    pub section_table: Vec<SectionEntry>,
    pub manifest: Manifest,
    pub footer: NestFooter,
}

impl<'a> NestView<'a> {
    pub fn from_bytes(data: &'a [u8]) -> crate::Result<Self> {
        if data.len() < NEST_HEADER_SIZE + NEST_FOOTER_SIZE {
            return Err(NestError::FileTruncated);
        }

        // ---- header -----------------------------------------------------
        let mut header = NestHeader::default();
        header
            .as_bytes_mut()
            .copy_from_slice(&data[..NEST_HEADER_SIZE]);

        if &header.magic != NEST_MAGIC {
            return Err(NestError::MagicMismatch {
                expected: *NEST_MAGIC,
                got: header.magic,
            });
        }
        if header.version_major != NEST_VERSION_MAJOR || header.version_minor > NEST_VERSION_MINOR {
            return Err(NestError::UnsupportedVersion(
                header.version_major,
                header.version_minor,
            ));
        }
        header.validate_checksum()?;

        if header.file_size as usize != data.len() {
            return Err(NestError::FileSizeMismatch {
                expected: header.file_size,
                got: data.len() as u64,
            });
        }

        // ---- section table ---------------------------------------------
        let section_table_offset = header.section_table_offset as usize;
        let section_table_count = header.section_table_count as usize;
        let section_table_end = section_table_offset
            .checked_add(
                section_table_count
                    .checked_mul(NEST_SECTION_ENTRY_SIZE)
                    .ok_or(NestError::SectionOffsetOutOfBounds {
                        section_id: 0,
                        offset: 0,
                    })?,
            )
            .ok_or(NestError::SectionOffsetOutOfBounds {
                section_id: 0,
                offset: 0,
            })?;
        if section_table_end > data.len().saturating_sub(NEST_FOOTER_SIZE) {
            return Err(NestError::FileTruncated);
        }

        let mut section_table = Vec::with_capacity(section_table_count);
        for i in 0..section_table_count {
            let off = section_table_offset + i * NEST_SECTION_ENTRY_SIZE;
            let mut entry = SectionEntry::new(0, 0, 0);
            entry
                .as_bytes_mut()
                .copy_from_slice(&data[off..off + NEST_SECTION_ENTRY_SIZE]);
            section_table.push(entry);
        }

        // ---- section bounds + alignment + encoding + checksums --------
        let body_end = data.len() - NEST_FOOTER_SIZE;
        for entry in &section_table {
            if entry.encoding != SECTION_ENCODING_RAW {
                return Err(NestError::UnsupportedSectionEncoding {
                    section_id: entry.section_id,
                    encoding: entry.encoding,
                });
            }
            if entry.offset % SECTION_ALIGNMENT != 0 {
                return Err(NestError::SectionMisaligned {
                    section_id: entry.section_id,
                    offset: entry.offset,
                    alignment: SECTION_ALIGNMENT,
                });
            }
            let start = entry.offset as usize;
            let end = start.checked_add(entry.size as usize).ok_or(
                NestError::SectionOffsetOutOfBounds {
                    section_id: entry.section_id,
                    offset: entry.offset,
                },
            )?;
            if end > body_end {
                return Err(NestError::SectionOffsetOutOfBounds {
                    section_id: entry.section_id,
                    offset: entry.offset,
                });
            }
            entry.validate_checksum(&data[start..end])?;
        }

        // ---- manifest ---------------------------------------------------
        let manifest_offset = header.manifest_offset as usize;
        let manifest_size = header.manifest_size as usize;
        if manifest_offset
            .checked_add(manifest_size)
            .map(|e| e > body_end)
            .unwrap_or(true)
        {
            return Err(NestError::FileTruncated);
        }
        let manifest_data = &data[manifest_offset..manifest_offset + manifest_size];
        let manifest: Manifest = serde_json::from_slice(manifest_data)?;
        manifest.validate()?;

        // ---- footer hash -----------------------------------------------
        let footer = NestFooter::from_bytes(&data[body_end..])?;
        let computed = NestFooter::compute_file_hash(&data[..body_end]);
        if computed != footer.file_hash {
            return Err(NestError::FooterHashMismatch);
        }

        // ---- coherence checks ------------------------------------------
        if manifest.embedding_dim != header.embedding_dim {
            return Err(NestError::ManifestInvalid(
                "manifest embedding_dim disagrees with header".into(),
            ));
        }
        if manifest.n_chunks != header.n_chunks {
            return Err(NestError::ManifestInvalid(
                "manifest n_chunks disagrees with header".into(),
            ));
        }

        let view = Self {
            data,
            header,
            section_table,
            manifest,
            footer,
        };

        // ---- required sections present ---------------------------------
        view.check_required_sections()?;

        // ---- embeddings size matches header ----------------------------
        view.validate_embeddings_layout()?;

        // ---- search_contract section agrees with manifest --------------
        view.validate_search_contract()?;

        Ok(view)
    }

    pub fn len(&self) -> usize {
        self.data.len()
    }

    pub fn is_empty(&self) -> bool {
        self.data.is_empty()
    }

    pub fn raw_bytes(&self) -> &[u8] {
        self.data
    }

    pub fn get_section_data(&self, section_id: u32) -> crate::Result<&'a [u8]> {
        for entry in &self.section_table {
            if entry.section_id == section_id {
                let start = entry.offset as usize;
                let end = start + entry.size as usize;
                return Ok(&self.data[start..end]);
            }
        }
        Err(NestError::SectionNotFound(section_id))
    }

    fn check_required_sections(&self) -> crate::Result<()> {
        for (id, name) in REQUIRED_SECTIONS {
            if !self.section_table.iter().any(|e| e.section_id == *id) {
                return Err(NestError::MissingRequiredSection(name));
            }
        }
        Ok(())
    }

    fn validate_embeddings_layout(&self) -> crate::Result<()> {
        let data = self.get_section_data(SECTION_EMBEDDINGS)?;
        let dim = self.header.embedding_dim as usize;
        let n = self.header.n_embeddings as usize;
        let expected = n.checked_mul(dim).and_then(|v| v.checked_mul(4)).ok_or(
            NestError::EmbeddingSizeMismatch {
                expected: 0,
                got: data.len(),
            },
        )?;
        if data.len() != expected {
            return Err(NestError::EmbeddingSizeMismatch {
                expected,
                got: data.len(),
            });
        }
        Ok(())
    }

    fn validate_search_contract(&self) -> crate::Result<()> {
        let bytes = self.get_section_data(SECTION_SEARCH_CONTRACT)?;
        let contract = decode_search_contract(bytes)?;
        if contract.metric != self.manifest.metric {
            return Err(NestError::UnsupportedMetric(format!(
                "section says {} but manifest says {}",
                contract.metric, self.manifest.metric
            )));
        }
        if contract.score_type != self.manifest.score_type {
            return Err(NestError::UnsupportedScoreType(format!(
                "section says {} but manifest says {}",
                contract.score_type, self.manifest.score_type
            )));
        }
        if contract.normalize != self.manifest.normalize {
            return Err(NestError::UnsupportedNormalize(format!(
                "section says {} but manifest says {}",
                contract.normalize, self.manifest.normalize
            )));
        }
        if contract.index_type != self.manifest.index_type {
            return Err(NestError::UnsupportedIndexType(format!(
                "section says {} but manifest says {}",
                contract.index_type, self.manifest.index_type
            )));
        }
        if contract.rerank_policy != self.manifest.rerank_policy {
            return Err(NestError::UnsupportedRerankPolicy(format!(
                "section says {} but manifest says {}",
                contract.rerank_policy, self.manifest.rerank_policy
            )));
        }
        Ok(())
    }

    /// Walk the embeddings section and reject any NaN/Inf value.
    pub fn validate_embeddings_values(&self) -> crate::Result<()> {
        self.validate_embeddings_layout()?;
        let data = self.get_section_data(SECTION_EMBEDDINGS)?;
        for chunk in data.chunks_exact(4) {
            let v = f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]);
            if v.is_nan() || v.is_infinite() {
                return Err(NestError::InvalidEmbeddingValue);
            }
        }
        Ok(())
    }

    /// Decode the `search_contract` section. Already validated to agree
    /// with the manifest at construction time.
    pub fn search_contract(&self) -> crate::Result<SearchContract> {
        decode_search_contract(self.get_section_data(SECTION_SEARCH_CONTRACT)?)
    }

    /// `sha256:<hex>` of the file as written, including the footer.
    pub fn file_hash_hex(&self) -> String {
        use sha2::{Digest, Sha256};
        let h = Sha256::digest(self.data);
        format!("sha256:{:x}", h)
    }

    /// `sha256:<hex>` of the canonical sections in the order fixed by spec
    /// (see `CANONICAL_SECTIONS`). Adding a section with a smaller numeric
    /// id cannot reshuffle this hash. All required sections were proved
    /// present at construction time, so in practice this never returns
    /// `Err`; the `Result` is preserved so callers never have to absorb
    /// a panic.
    pub fn content_hash_hex(&self) -> crate::Result<String> {
        use sha2::{Digest, Sha256};
        let mut h = Sha256::new();
        for (id, name) in CANONICAL_SECTIONS {
            let data = self.get_section_data(*id)?;
            // Domain-separate by name length + name bytes so hashes for
            // different sections cannot collide via concatenation.
            h.update((name.len() as u32).to_le_bytes());
            h.update(name.as_bytes());
            h.update((data.len() as u64).to_le_bytes());
            h.update(data);
        }
        Ok(format!("sha256:{:x}", h.finalize()))
    }
}
