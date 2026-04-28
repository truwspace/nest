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

mod canonical;
mod chunk_ids;
mod codec;
mod contract;
mod provenance;
mod spans;

pub use canonical::{decode_chunks_canonical, encode_chunks_canonical};
pub use chunk_ids::{decode_chunk_ids, encode_chunk_ids};
pub use contract::{SearchContract, decode_search_contract, encode_search_contract};
pub use provenance::{decode_provenance, encode_provenance};
pub use spans::{OriginalSpan, decode_chunks_original_spans, encode_chunks_original_spans};
