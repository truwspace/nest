/// .nest binary file layout (v1)
///
/// File structure:
/// ```text
/// [0 .. 128)          NestHeader (128 bytes)
/// [header.section_table_offset .. +count*32)  SectionTable (32 bytes per entry)
/// [manifest_offset .. +manifest_size)        Manifest JSON (JCS canonical)
/// [sections ...]                             Chunks, embeddings, ids, provenance, etc.
/// [file_size-40 .. file_size)               Footer (40 bytes)
/// ```

use crate::error::NestError;
use sha2::{Digest, Sha256};

pub const NEST_MAGIC: &[u8; 4] = b"NEST";
pub const NEST_VERSION_MAJOR: u16 = 1;
pub const NEST_VERSION_MINOR: u16 = 0;
pub const NEST_HEADER_SIZE: usize = 128;
pub const NEST_SECTION_ENTRY_SIZE: usize = 32;
pub const NEST_FOOTER_SIZE: usize = 40;

/// Section IDs (application-defined, format only validates presence/checksum)
pub const SECTION_CHUNKS: u32 = 0x01;
pub const SECTION_EMBEDDINGS: u32 = 0x02;
pub const SECTION_CHUNK_IDS: u32 = 0x03;
pub const SECTION_PROVENANCE: u32 = 0x04;
pub const SECTION_SEARCH_CONTRACT: u32 = 0x05;

/// Fixed-size binary header (128 bytes, no padding issues because all fields sized)
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
        if &hash[..8] != &self.header_checksum[..] {
            return Err(NestError::InvalidHeaderChecksum);
        }
        Ok(())
    }

    pub fn as_bytes(&self) -> &[u8] {
        // Safe: NestHeader is repr(C) plain-old-data
        let size = std::mem::size_of::<Self>();
        unsafe { std::slice::from_raw_parts(self as *const _ as *const u8, size) }
    }

    pub fn as_bytes_mut(&mut self) -> &mut [u8] {
        let size = std::mem::size_of::<Self>();
        unsafe { std::slice::from_raw_parts_mut(self as *mut _ as *mut u8, size) }
    }

    fn as_bytes_without_checksum(&self) -> Vec<u8> {
        let bytes = self.as_bytes();
        let mut v = bytes[..72].to_vec(); // 0..72 before header_checksum
        v.extend_from_slice(&bytes[80..]); // 80..128 reserved after checksum
        v
    }
}

/// Section table entry (32 bytes)
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SectionEntry {
    pub section_id: u32,
    pub reserved: u32,
    pub offset: u64,
    pub size: u64,
    pub checksum: [u8; 8],
    pub reserved2: [u8; 8],
}

impl SectionEntry {
    pub fn new(section_id: u32, offset: u64, size: u64) -> Self {
        Self {
            section_id,
            reserved: 0,
            offset,
            size,
            checksum: [0; 8],
            reserved2: [0; 8],
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
        if &hash[..8] != &self.checksum[..] {
            return Err(NestError::SectionChecksumMismatch(self.section_id));
        }
        Ok(())
    }
}

/// Footer (40 bytes)
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

    pub fn compute_file_hash(data_without_footer: &[u8]) -> [u8; 32] {
        let hash = Sha256::digest(data_without_footer);
        let mut out = [0u8; 32];
        out.copy_from_slice(&hash[..]);
        out
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
}
