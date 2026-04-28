//! `chunks_original_spans` section (`SECTION_CHUNKS_ORIGINAL_SPANS = 0x03`).
//! `(source_uri, byte_start, byte_end)` per chunk — the offset into the
//! original source document the chunk text came from. Required for
//! citation resolution.

use super::codec::{Cursor, read_prefix, write_lp_str, write_prefix};
use crate::error::NestError;
use crate::layout::SECTION_CHUNKS_ORIGINAL_SPANS;

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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn roundtrip() {
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
}
