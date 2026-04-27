//! Section payload formats (v1).
//!
//! Each non-binary section starts with a 12-byte header:
//!
//! ```text
//! [0..4)   u32 version  (LE) — currently 1
//! [4..12)  u64 count    (LE) — number of entries
//! ```
//!
//! Then a payload that depends on the section. Embeddings use a different
//! shape (no per-entry header — dim/count come from the file header).
//!
//! All multi-byte integers are little-endian. Strings are raw UTF-8 bytes
//! prefixed by a u32 length (no NUL terminators).

use crate::error::NestError;
use crate::layout::{
    SECTION_CHUNK_IDS, SECTION_CHUNKS_CANONICAL, SECTION_CHUNKS_ORIGINAL_SPANS,
    SECTION_PAYLOAD_PREFIX_SIZE, SECTION_PAYLOAD_VERSION, SECTION_PROVENANCE,
    SECTION_SEARCH_CONTRACT,
};

// ---------------------------------------------------------------------------
// Encoding helpers
// ---------------------------------------------------------------------------

fn write_prefix(buf: &mut Vec<u8>, count: u64) {
    buf.extend_from_slice(&SECTION_PAYLOAD_VERSION.to_le_bytes());
    buf.extend_from_slice(&count.to_le_bytes());
}

fn write_lp_str(buf: &mut Vec<u8>, s: &str) -> crate::Result<()> {
    let bytes = s.as_bytes();
    let len = u32::try_from(bytes.len())
        .map_err(|_| NestError::InvalidInput(format!("string too long: {} bytes", bytes.len())))?;
    buf.extend_from_slice(&len.to_le_bytes());
    buf.extend_from_slice(bytes);
    Ok(())
}

// ---------------------------------------------------------------------------
// Decoding helpers
// ---------------------------------------------------------------------------

struct Cursor<'a> {
    data: &'a [u8],
    pos: usize,
    section_id: u32,
}

impl<'a> Cursor<'a> {
    fn new(data: &'a [u8], section_id: u32) -> Self {
        Self {
            data,
            pos: 0,
            section_id,
        }
    }

    fn malformed(&self, reason: impl Into<String>) -> NestError {
        NestError::MalformedSectionPayload {
            section_id: self.section_id,
            reason: reason.into(),
        }
    }

    fn read_bytes(&mut self, n: usize) -> Result<&'a [u8], NestError> {
        if self.pos + n > self.data.len() {
            return Err(self.malformed(format!(
                "want {} bytes at offset {}, have {}",
                n,
                self.pos,
                self.data.len()
            )));
        }
        let out = &self.data[self.pos..self.pos + n];
        self.pos += n;
        Ok(out)
    }

    fn read_u32(&mut self) -> Result<u32, NestError> {
        let b = self.read_bytes(4)?;
        Ok(u32::from_le_bytes([b[0], b[1], b[2], b[3]]))
    }

    fn read_u64(&mut self) -> Result<u64, NestError> {
        let b = self.read_bytes(8)?;
        Ok(u64::from_le_bytes([
            b[0], b[1], b[2], b[3], b[4], b[5], b[6], b[7],
        ]))
    }

    fn read_lp_str(&mut self) -> Result<String, NestError> {
        let len = self.read_u32()? as usize;
        let bytes = self.read_bytes(len)?;
        std::str::from_utf8(bytes)
            .map(|s| s.to_string())
            .map_err(|e| self.malformed(format!("invalid utf-8: {}", e)))
    }

    fn finish(self) -> Result<(), NestError> {
        if self.pos != self.data.len() {
            return Err(self.malformed(format!(
                "trailing bytes: consumed {} of {}",
                self.pos,
                self.data.len()
            )));
        }
        Ok(())
    }
}

fn read_prefix(c: &mut Cursor) -> Result<u64, NestError> {
    if c.data.len() < SECTION_PAYLOAD_PREFIX_SIZE {
        return Err(c.malformed(format!(
            "payload shorter than {} byte prefix",
            SECTION_PAYLOAD_PREFIX_SIZE
        )));
    }
    let version = c.read_u32()?;
    if version != SECTION_PAYLOAD_VERSION {
        return Err(NestError::UnsupportedSectionVersion {
            section_id: c.section_id,
            version,
        });
    }
    c.read_u64()
}

// ---------------------------------------------------------------------------
// chunk_ids (SECTION_CHUNK_IDS)
// ---------------------------------------------------------------------------

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

// ---------------------------------------------------------------------------
// chunks_canonical (SECTION_CHUNKS_CANONICAL)
// ---------------------------------------------------------------------------

pub fn encode_chunks_canonical(texts: &[String]) -> crate::Result<Vec<u8>> {
    let mut buf = Vec::new();
    write_prefix(&mut buf, texts.len() as u64);
    for t in texts {
        write_lp_str(&mut buf, t)?;
    }
    Ok(buf)
}

pub fn decode_chunks_canonical(data: &[u8], expected_count: usize) -> crate::Result<Vec<String>> {
    let mut c = Cursor::new(data, SECTION_CHUNKS_CANONICAL);
    let count = read_prefix(&mut c)? as usize;
    if count != expected_count {
        return Err(NestError::SectionCountMismatch {
            section_id: SECTION_CHUNKS_CANONICAL,
            expected: expected_count,
            got: count,
        });
    }
    let mut out = Vec::with_capacity(count);
    for _ in 0..count {
        out.push(c.read_lp_str()?);
    }
    c.finish()?;
    Ok(out)
}

// ---------------------------------------------------------------------------
// chunks_original_spans (SECTION_CHUNKS_ORIGINAL_SPANS)
// ---------------------------------------------------------------------------

#[derive(Clone, Debug, PartialEq)]
pub struct OriginalSpan {
    pub source_uri: String,
    pub byte_start: u64,
    pub byte_end: u64,
}

pub fn encode_chunks_original_spans(spans: &[OriginalSpan]) -> crate::Result<Vec<u8>> {
    let mut buf = Vec::new();
    write_prefix(&mut buf, spans.len() as u64);
    for s in spans {
        write_lp_str(&mut buf, &s.source_uri)?;
        buf.extend_from_slice(&s.byte_start.to_le_bytes());
        buf.extend_from_slice(&s.byte_end.to_le_bytes());
    }
    Ok(buf)
}

pub fn decode_chunks_original_spans(
    data: &[u8],
    expected_count: usize,
) -> crate::Result<Vec<OriginalSpan>> {
    let mut c = Cursor::new(data, SECTION_CHUNKS_ORIGINAL_SPANS);
    let count = read_prefix(&mut c)? as usize;
    if count != expected_count {
        return Err(NestError::SectionCountMismatch {
            section_id: SECTION_CHUNKS_ORIGINAL_SPANS,
            expected: expected_count,
            got: count,
        });
    }
    let mut out = Vec::with_capacity(count);
    for _ in 0..count {
        let source_uri = c.read_lp_str()?;
        let byte_start = c.read_u64()?;
        let byte_end = c.read_u64()?;
        out.push(OriginalSpan {
            source_uri,
            byte_start,
            byte_end,
        });
    }
    c.finish()?;
    Ok(out)
}

// ---------------------------------------------------------------------------
// provenance (SECTION_PROVENANCE)
//
// Versioned single-blob JSON. We keep this flexible for ingestion metadata
// without committing to a strict schema in v1.
// ---------------------------------------------------------------------------

pub fn encode_provenance(value: &serde_json::Value) -> crate::Result<Vec<u8>> {
    let json = serde_json::to_vec(value)?;
    let json_len = u32::try_from(json.len())
        .map_err(|_| NestError::InvalidInput("provenance JSON too large".into()))?;
    let mut buf = Vec::with_capacity(SECTION_PAYLOAD_PREFIX_SIZE + json.len());
    buf.extend_from_slice(&SECTION_PAYLOAD_VERSION.to_le_bytes());
    buf.extend_from_slice(&(json_len as u64).to_le_bytes());
    buf.extend_from_slice(&json);
    Ok(buf)
}

pub fn decode_provenance(data: &[u8]) -> crate::Result<serde_json::Value> {
    let mut c = Cursor::new(data, SECTION_PROVENANCE);
    if c.data.len() < SECTION_PAYLOAD_PREFIX_SIZE {
        return Err(c.malformed("payload shorter than prefix"));
    }
    let version = c.read_u32()?;
    if version != SECTION_PAYLOAD_VERSION {
        return Err(NestError::UnsupportedSectionVersion {
            section_id: SECTION_PROVENANCE,
            version,
        });
    }
    let json_len = c.read_u64()? as usize;
    let json = c.read_bytes(json_len)?;
    let value: serde_json::Value = serde_json::from_slice(json).map_err(NestError::Json)?;
    c.finish()?;
    Ok(value)
}

// ---------------------------------------------------------------------------
// search_contract (SECTION_SEARCH_CONTRACT)
//
// A small JSON document with the runtime contract. Same versioned shape
// as provenance.
// ---------------------------------------------------------------------------

#[derive(Clone, Debug, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct SearchContract {
    pub metric: String,
    pub score_type: String,
    pub normalize: String,
    pub index_type: String,
    pub rerank_policy: String,
}

pub fn encode_search_contract(contract: &SearchContract) -> crate::Result<Vec<u8>> {
    let json = serde_json::to_vec(contract)?;
    let json_len = u32::try_from(json.len())
        .map_err(|_| NestError::InvalidInput("search_contract JSON too large".into()))?;
    let mut buf = Vec::with_capacity(SECTION_PAYLOAD_PREFIX_SIZE + json.len());
    buf.extend_from_slice(&SECTION_PAYLOAD_VERSION.to_le_bytes());
    buf.extend_from_slice(&(json_len as u64).to_le_bytes());
    buf.extend_from_slice(&json);
    Ok(buf)
}

pub fn decode_search_contract(data: &[u8]) -> crate::Result<SearchContract> {
    let mut c = Cursor::new(data, SECTION_SEARCH_CONTRACT);
    if c.data.len() < SECTION_PAYLOAD_PREFIX_SIZE {
        return Err(c.malformed("payload shorter than prefix"));
    }
    let version = c.read_u32()?;
    if version != SECTION_PAYLOAD_VERSION {
        return Err(NestError::UnsupportedSectionVersion {
            section_id: SECTION_SEARCH_CONTRACT,
            version,
        });
    }
    let json_len = c.read_u64()? as usize;
    let json = c.read_bytes(json_len)?;
    let contract: SearchContract = serde_json::from_slice(json).map_err(NestError::Json)?;
    c.finish()?;
    Ok(contract)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn chunk_ids_roundtrip() {
        let ids = vec!["sha256:aaa".to_string(), "sha256:bbb".to_string()];
        let bytes = encode_chunk_ids(&ids).unwrap();
        let back = decode_chunk_ids(&bytes, 2).unwrap();
        assert_eq!(ids, back);
    }

    #[test]
    fn chunk_ids_count_mismatch() {
        let ids = vec!["a".to_string()];
        let bytes = encode_chunk_ids(&ids).unwrap();
        let err = decode_chunk_ids(&bytes, 5).unwrap_err();
        assert!(matches!(err, NestError::SectionCountMismatch { .. }));
    }

    #[test]
    fn chunks_canonical_roundtrip() {
        let texts = vec!["primeiro".to_string(), "segundo".to_string()];
        let bytes = encode_chunks_canonical(&texts).unwrap();
        let back = decode_chunks_canonical(&bytes, 2).unwrap();
        assert_eq!(texts, back);
    }

    #[test]
    fn original_spans_roundtrip() {
        let spans = vec![
            OriginalSpan {
                source_uri: "doc.txt".into(),
                byte_start: 0,
                byte_end: 10,
            },
            OriginalSpan {
                source_uri: "doc.txt".into(),
                byte_start: 10,
                byte_end: 25,
            },
        ];
        let bytes = encode_chunks_original_spans(&spans).unwrap();
        let back = decode_chunks_original_spans(&bytes, 2).unwrap();
        assert_eq!(spans, back);
    }

    #[test]
    fn search_contract_roundtrip() {
        let c = SearchContract {
            metric: "ip".into(),
            score_type: "cosine".into(),
            normalize: "l2".into(),
            index_type: "exact".into(),
            rerank_policy: "none".into(),
        };
        let bytes = encode_search_contract(&c).unwrap();
        let back = decode_search_contract(&bytes).unwrap();
        assert_eq!(c, back);
    }

    #[test]
    fn provenance_roundtrip() {
        let v = serde_json::json!({"source": "demo", "ingested_at": "2026-01-01"});
        let bytes = encode_provenance(&v).unwrap();
        let back = decode_provenance(&bytes).unwrap();
        assert_eq!(v, back);
    }

    #[test]
    fn rejects_unsupported_section_version() {
        // Forge a chunk_ids payload with version 99
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
