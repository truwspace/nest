//! `chunk_ids` section (`SECTION_CHUNK_IDS = 0x01`). Length-prefixed
//! UTF-8 strings of the form `sha256:<64 hex>` — one per chunk.

use super::codec::{Cursor, read_prefix, write_lp_str, write_prefix};
use crate::error::NestError;
use crate::layout::SECTION_CHUNK_IDS;

pub fn encode_chunk_ids(ids: &[String]) -> crate::Result<Vec<u8>> {
    let mut buf = Vec::new();
    write_prefix(&mut buf, ids.len() as u64);
    for id in ids {
        write_lp_str(&mut buf, id)?;
    }
    Ok(buf)
}

pub fn decode_chunk_ids(data: &[u8], expected_count: usize) -> crate::Result<Vec<String>> {
    let mut c = Cursor::new(data, SECTION_CHUNK_IDS);
    let count = read_prefix(&mut c)? as usize;
    if count != expected_count {
        return Err(NestError::SectionCountMismatch {
            section_id: SECTION_CHUNK_IDS,
            expected: expected_count,
            got: count,
        });
    }
    let mut ids = Vec::with_capacity(count);
    for _ in 0..count {
        ids.push(c.read_lp_str()?);
    }
    c.finish()?;
    Ok(ids)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::layout::SECTION_CHUNK_IDS;

    #[test]
    fn roundtrip() {
        let ids = vec!["sha256:aaa".to_string(), "sha256:bbb".to_string()];
        let bytes = encode_chunk_ids(&ids).unwrap();
        let back = decode_chunk_ids(&bytes, 2).unwrap();
        assert_eq!(ids, back);
    }

    #[test]
    fn count_mismatch() {
        let ids = vec!["a".to_string()];
        let bytes = encode_chunk_ids(&ids).unwrap();
        let err = decode_chunk_ids(&bytes, 5).unwrap_err();
        assert!(matches!(err, NestError::SectionCountMismatch { .. }));
    }

    #[test]
    fn rejects_unsupported_version() {
        let mut buf = Vec::new();
        buf.extend_from_slice(&99u32.to_le_bytes());
        buf.extend_from_slice(&0u64.to_le_bytes());
        let err = decode_chunk_ids(&buf, 0).unwrap_err();
        assert!(matches!(
            err,
            NestError::UnsupportedSectionVersion {
                section_id: SECTION_CHUNK_IDS,
                version: 99,
            }
        ));
    }
}
