//! .nest binary file layout (v1)
//!
//! File structure:
//! ```text
//! [0 .. 128)                  NestHeader (128 bytes)
//! [128 .. 128+count*32)       SectionTable (32 bytes per entry)
//! [manifest_offset .. ...)    Manifest JSON (JCS canonical)
//! [sections ...]              Required sections, each starting at a
//!                             64-byte aligned offset; padding before
//!                             each section is zero and is NOT part of
//!                             the section's checksum
//! [file_size-40 .. file_size) Footer (40 bytes)
//! ```
//!
//! All multi-byte integers are little-endian, unsigned unless noted.

use crate::error::NestError;
use sha2::{Digest, Sha256};

pub const NEST_MAGIC: &[u8; 4] = b"NEST";
pub const NEST_VERSION_MAJOR: u16 = 1;
pub const NEST_VERSION_MINOR: u16 = 0;
pub const NEST_HEADER_SIZE: usize = 128;
pub const NEST_SECTION_ENTRY_SIZE: usize = 32;
pub const NEST_FOOTER_SIZE: usize = 40;

/// Every section's `offset` is aligned to this many bytes. Padding
/// before each section is zero and is NOT covered by the section's
/// checksum. Chosen to match common SIMD widths so embeddings can be
/// loaded directly from mmap.
pub const SECTION_ALIGNMENT: u64 = 64;

/// Round `n` up to the next multiple of `a`. `a` must be a power of two.
#[inline]
pub fn align_up(n: u64, a: u64) -> u64 {
    debug_assert!(a.is_power_of_two(), "alignment must be a power of two");
    (n + a - 1) & !(a - 1)
}

/// Section payload encoding.
///
/// - `0 = raw`: payload is the canonical bytes as the reader consumes them.
///   Used for embeddings (float32) and any non-compressed metadata section.
/// - `1 = zstd`: payload is zstd-compressed canonical bytes. Only valid for
///   non-embedding sections; the reader transparently decompresses.
/// - `2 = float16`: payload is `n * dim * 2` bytes of f16 LE; requires the
///   manifest to declare `dtype = "float16"`. Only valid for the embeddings
///   section.
/// - `3 = int8`: payload is the int8 quantized embeddings section (per-vector
///   f32 scales followed by i8 vectors); requires `dtype = "int8"`. Only
///   valid for the embeddings section.
///
/// A reader rejects unknown encodings with `UnsupportedSectionEncoding`.
pub const SECTION_ENCODING_RAW: u32 = 0;
pub const SECTION_ENCODING_ZSTD: u32 = 1;
pub const SECTION_ENCODING_FLOAT16: u32 = 2;
pub const SECTION_ENCODING_INT8: u32 = 3;

/// Format version of the binary layout. Bumped when the on-disk
/// container changes (header/footer/section table layout).
pub const NEST_FORMAT_VERSION: u32 = 1;

/// Schema version of the manifest/contract. Bumped when manifest
/// fields or required section semantics change.
pub const NEST_SCHEMA_VERSION: u32 = 1;

// Section IDs. The first six are required (v1 contract); the rest are
// optional and only present when the manifest's `capabilities` declare
// them.
pub const SECTION_CHUNK_IDS: u32 = 0x01;
pub const SECTION_CHUNKS_CANONICAL: u32 = 0x02;
pub const SECTION_CHUNKS_ORIGINAL_SPANS: u32 = 0x03;
pub const SECTION_EMBEDDINGS: u32 = 0x04;
pub const SECTION_PROVENANCE: u32 = 0x05;
pub const SECTION_SEARCH_CONTRACT: u32 = 0x06;
pub const SECTION_HNSW_INDEX: u32 = 0x07;
pub const SECTION_BM25_INDEX: u32 = 0x08;

/// Canonical order for content_hash. Sorted alphabetically by name; this
/// order is fixed by spec so adding new section IDs cannot reshuffle the
/// hash. Keep this list and section IDs in sync.
pub const CANONICAL_SECTIONS: &[(u32, &str)] = &[
    (SECTION_CHUNK_IDS, "chunk_ids"),
    (SECTION_CHUNKS_CANONICAL, "chunks_canonical"),
    (SECTION_CHUNKS_ORIGINAL_SPANS, "chunks_original_spans"),
    (SECTION_EMBEDDINGS, "embeddings"),
    (SECTION_PROVENANCE, "provenance"),
    (SECTION_SEARCH_CONTRACT, "search_contract"),
];

/// Required sections for a v1 .nest file. A reader rejects any file
/// missing one of these with `MissingRequiredSection`.
pub const REQUIRED_SECTIONS: &[(u32, &str)] = CANONICAL_SECTIONS;

/// Optional sections — present when their corresponding capability is
/// advertised in the manifest. They do NOT participate in content_hash
/// (which is over the canonical six only) so adding an optional section
/// to a corpus does not invalidate citations.
pub const OPTIONAL_SECTIONS: &[(u32, &str)] = &[
    (SECTION_HNSW_INDEX, "hnsw_index"),
    (SECTION_BM25_INDEX, "bm25_index"),
];

pub fn section_name(id: u32) -> Option<&'static str> {
    CANONICAL_SECTIONS
        .iter()
        .chain(OPTIONAL_SECTIONS.iter())
        .find(|(sid, _)| *sid == id)
        .map(|(_, name)| *name)
}

/// Common prefix for all internal section payloads (12 bytes):
///   u32 version (LE)
///   u64 entry_count (LE)
pub const SECTION_PAYLOAD_PREFIX_SIZE: usize = 12;
pub const SECTION_PAYLOAD_VERSION: u32 = 1;

/// Fixed-size binary header (128 bytes).
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct NestHeader {
    pub magic: [u8; 4],
    pub version_major: u16,
    pub version_minor: u16,
    pub flags: u32,
    pub embedding_dim: u32,
    pub n_chunks: u64,
    pub n_embeddings: u64,
    pub file_size: u64,
    pub section_table_offset: u64,
    pub section_table_count: u64,
    pub manifest_offset: u64,
    pub manifest_size: u64,
    pub header_checksum: [u8; 8],
    pub reserved: [u8; 48],
}

impl NestHeader {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        embedding_dim: u32,
        n_chunks: u64,
        n_embeddings: u64,
        file_size: u64,
        section_table_offset: u64,
        section_table_count: u64,
        manifest_offset: u64,
        manifest_size: u64,
    ) -> Self {
        let mut h = Self {
            magic: *NEST_MAGIC,
            version_major: NEST_VERSION_MAJOR,
            version_minor: NEST_VERSION_MINOR,
            flags: 0,
            embedding_dim,
            n_chunks,
            n_embeddings,
            file_size,
            section_table_offset,
            section_table_count,
            manifest_offset,
            manifest_size,
            header_checksum: [0; 8],
            reserved: [0; 48],
        };
        h.compute_checksum();
        h
    }

    pub fn compute_checksum(&mut self) {
        let bytes = self.as_bytes_without_checksum();
        let hash = Sha256::digest(&bytes);
        self.header_checksum.copy_from_slice(&hash[..8]);
    }

    pub fn validate_checksum(&self) -> crate::Result<()> {
        let mut tmp = *self;
        tmp.header_checksum = [0; 8];
        let bytes = tmp.as_bytes_without_checksum();
        let hash = Sha256::digest(&bytes);
        if hash[..8] != self.header_checksum[..] {
            return Err(NestError::InvalidHeaderChecksum);
        }
        Ok(())
    }

    pub fn as_bytes(&self) -> &[u8] {
        let size = std::mem::size_of::<Self>();
        unsafe { std::slice::from_raw_parts(self as *const _ as *const u8, size) }
    }

    pub fn as_bytes_mut(&mut self) -> &mut [u8] {
        let size = std::mem::size_of::<Self>();
        unsafe { std::slice::from_raw_parts_mut(self as *mut _ as *mut u8, size) }
    }

    fn as_bytes_without_checksum(&self) -> Vec<u8> {
        let bytes = self.as_bytes();
        let mut v = bytes[..72].to_vec();
        v.extend_from_slice(&bytes[80..]);
        v
    }
}

impl Default for NestHeader {
    fn default() -> Self {
        Self::new(0, 0, 0, 128, 128, 0, 128, 0)
    }
}

/// Section table entry (32 bytes).
///
/// `offset` is 64-byte aligned (see `SECTION_ALIGNMENT`). `size` is the
/// length of the actual payload, NOT including the trailing padding.
/// `encoding` declares the on-disk payload format; v1 only supports
/// `SECTION_ENCODING_RAW` (0).
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SectionEntry {
    pub section_id: u32,
    pub encoding: u32,
    pub offset: u64,
    pub size: u64,
    pub checksum: [u8; 8],
}

impl SectionEntry {
    pub fn new(section_id: u32, offset: u64, size: u64) -> Self {
        Self {
            section_id,
            encoding: SECTION_ENCODING_RAW,
            offset,
            size,
            checksum: [0; 8],
        }
    }

    pub fn as_bytes(&self) -> &[u8] {
        let size = std::mem::size_of::<Self>();
        unsafe { std::slice::from_raw_parts(self as *const _ as *const u8, size) }
    }

    pub fn as_bytes_mut(&mut self) -> &mut [u8] {
        let size = std::mem::size_of::<Self>();
        unsafe { std::slice::from_raw_parts_mut(self as *mut _ as *mut u8, size) }
    }

    pub fn compute_checksum(&mut self, data: &[u8]) {
        let hash = Sha256::digest(data);
        self.checksum.copy_from_slice(&hash[..8]);
    }

    pub fn validate_checksum(&self, data: &[u8]) -> crate::Result<()> {
        let hash = Sha256::digest(data);
        if hash[..8] != self.checksum[..] {
            return Err(NestError::SectionChecksumMismatch(self.section_id));
        }
        Ok(())
    }
}

/// Footer (40 bytes).
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct NestFooter {
    pub footer_size: u64,
    pub file_hash: [u8; 32],
}

impl NestFooter {
    pub fn new(file_hash: [u8; 32]) -> Self {
        Self {
            footer_size: NEST_FOOTER_SIZE as u64,
            file_hash,
        }
    }

    pub fn as_bytes(&self) -> &[u8] {
        let size = std::mem::size_of::<Self>();
        unsafe { std::slice::from_raw_parts(self as *const _ as *const u8, size) }
    }

    pub fn as_bytes_mut(&mut self) -> &mut [u8] {
        let size = std::mem::size_of::<Self>();
        unsafe { std::slice::from_raw_parts_mut(self as *mut _ as *mut u8, size) }
    }

    pub fn compute_file_hash(data_without_footer: &[u8]) -> [u8; 32] {
        let hash = Sha256::digest(data_without_footer);
        let mut out = [0u8; 32];
        out.copy_from_slice(&hash[..]);
        out
    }

    pub fn from_bytes(bytes: &[u8]) -> crate::Result<Self> {
        if bytes.len() < std::mem::size_of::<Self>() {
            return Err(NestError::UnexpectedEof);
        }
        let mut footer = NestFooter::new([0; 32]);
        footer.as_bytes_mut().copy_from_slice(bytes);
        Ok(footer)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn header_size_is_128() {
        assert_eq!(std::mem::size_of::<NestHeader>(), 128);
    }

    #[test]
    fn section_entry_size_is_32() {
        assert_eq!(std::mem::size_of::<SectionEntry>(), 32);
    }

    #[test]
    fn footer_size_is_40() {
        assert_eq!(std::mem::size_of::<NestFooter>(), 40);
    }

    #[test]
    fn header_roundtrip_checksum() {
        let mut h = NestHeader::new(384, 100, 100, 1024, 128, 5, 288, 200);
        assert!(h.validate_checksum().is_ok());
        h.n_chunks = 99;
        assert!(h.validate_checksum().is_err());
    }

    #[test]
    fn canonical_sections_are_alphabetical_by_name() {
        let names: Vec<&str> = CANONICAL_SECTIONS.iter().map(|(_, n)| *n).collect();
        let mut sorted = names.clone();
        sorted.sort_unstable();
        assert_eq!(names, sorted);
    }

    #[test]
    fn section_name_lookup() {
        assert_eq!(section_name(SECTION_CHUNK_IDS), Some("chunk_ids"));
        assert_eq!(section_name(SECTION_EMBEDDINGS), Some("embeddings"));
        assert_eq!(section_name(0xFFFF), None);
    }
}
