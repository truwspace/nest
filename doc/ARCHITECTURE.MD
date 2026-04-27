nest

Portable, hash-verified, mmap-friendly container for semantic knowledge: canonical chunks, their float32 embeddings, original-document spans, and a runtime contract that pins every search-time decision.

.nest v1 is frozen. The byte-by-byte specification is in SPEC.md. The Rust implementation in crates/nest-format is the canonical encoder/decoder.

Principles

Reader-first. Every read path validates magic, header checksum, section checksums, footer hash, manifest contract, required-section presence, and embedding values. Failures return typed errors, never panics.

Hash everything. Header, every section, the file as a whole, and a content_hash over the canonical sections. No silent corruption.

Recall before throughput. v1 ships flat exact search only. ANN arrives in a future format version.

Reproducible. With reproducible=True, two builds with identical inputs produce byte-identical files. The manifest created timestamp is overridden to 1970-01-01T00:00:00Z. The proofen JSON is not rewritten; callers must keep it deterministic for bit-for-bit equality.

Citable. Every chunk has a stable chunk_id derived from a domain-separated SHA-256 preimage. Every hit is resolvable via nest://<content_hash>/<chunk_id>.

Binary format

All multi-byte integers are little-endian unsigned. SHA-256 is the only hash. Hex is lowercase. Format: sha256:<64 hex chars>. Sections are 64-byte aligned (padding is zero, not checksummed). Strings are length-prefixed UTF-8 (u32 LE length, then bytes, no NUL terminator).

File layout:

  [0 .. 128)                  NestHeader (128 bytes)
  [128 .. 128+count*32)        SectionTable (32 bytes per entry)
  [manifest_offset ..)         Manifest (JCS canonical JSON, no whitespace)
  [sections ...]               6 required sections, each 64-byte aligned
  [file_size-40 .. file_size)   Footer (40 bytes)

Header (NestHeader, 128 bytes):

  Offset  Size  Field
  0       4     magic                     ASCII "NEST" (0x4E455354)
  4       2     version_major             1
  6       2     version_minor             0
  8       4     flags                     0 in v1
  12      4     embedding_dim             > 0
  16      8     n_chunks
  24      8     n_embeddings              always == n_chunks in v1
  32      8     file_size                 total bytes
  40      8     section_table_offset      always 128
  48      8     section_table_count
  56      8     manifest_offset
  64      8     manifest_size
  72      8     header_checksum           SHA-256(header[0..72] ++ header[80..128]), first 8 bytes
  80      48    reserved                  all zero in v1

Section table entry (SectionEntry, 32 bytes):

  Offset  Size  Field
  0       4     section_id               one of 0x01..0x06
  4       4     encoding                 0 = raw (only value in v1)
  8       8     offset                   must be 64-byte aligned
  16      8     size                     payload length, excludes padding
  24      8     checksum                 SHA-256(payload), first 8 bytes

Required sections (v1, sorted by section_id in the table, alphabetical by name for content_hash):

  0x01  chunk_ids                length-prefixed UTF-8 strings (sha256:<64 hex>)
  0x02  chunks_canonical         length-prefixed UTF-8 strings
  0x03  chunks_original_spans    per-chunk (source_uri, byte_start, byte_end)
  0x04  embeddings               raw float32 LE, shape [n_chunks, embedding_dim], L2-normalized, no NaN/Inf
  0x05  provenance               versioned JSON blob (free-form, deterministic)
  0x06  search_contract          versioned JSON: {metric:"ip", score_type:"cosine", normalize:"l2", index_type:"exact", rerank_policy:"none"}

Every section except embeddings (0x04) starts with a 12-byte prefix: u32 version (1) + u64 count. Embeddings has no prefix; payload is exactly n_chunks * embedding_dim * 4 bytes.

Footer (NestFooter, 40 bytes):

  Offset  Size  Field
  0       8     footer_size              always 40
  8       32    file_hash                SHA-256(file[0..file_size-40])

chunk_id derivation

  preimage = "nest:chunk_id:v1\n"
             u32 LE len(canonical_text)  bytes canonical_text
             u32 LE len(source_uri)       bytes source_uri
             u64 LE byte_start
             u64 LE byte_end
             u32 LE len(chunker_version)  bytes chunker_version

  chunk_id = sha256:<hex( SHA-256(preimage) )>

content_hash

  Hash over all 6 canonical sections in alphabetical order by name. For each section: u32 LE len(name) + name bytes + u64 LE len(data) + data. Padding is not included. Stable across rebuilds even if wrapper metadata changes.

Citation URI

  nest://<content_hash>/<chunk_id>

  Both content_hash and chunk_id are full sha256:<hex> strings. Resolution requires content_hash match and chunk_id lookup.

Manifest

  JCS canonical JSON. Required fields: format_version (1), schema_version (1), embedding_model, embedding_dim, n_chunks, dtype ("float32"), metric ("ip"), score_type ("cosine"), normalize ("l2"), index_type ("exact"), rerank_policy ("none"), model_hash (sha256:<64 hex>), chunker_version, capabilities {supports_exact:true, supports_ann:false, supports_bm25:false, supports_citations:true, supports_reproducible_build:true}.

  Optional: title, version, created, description, authors, license. Extra keys in a BTreeMap serialized after known fields.

  When reproducible: created is overridden to 1970-01-01T00:00:00Z.

Search contract (v1)

  Flat exact search. Query is L2-normalized internally. Score is real cosine. Top-k is stable sort by score descending, ties broken by index ascending. recall == 1.0 always. truncated == true iff k < n_embeddings. Validations: k > 0, query non-empty, len(query) == embedding_dim, no NaN/Inf in query, norm > 0.

  SearchHit carries: chunk_id, score (f32), score_type ("cosine"), source_uri, offset_start, offset_end, embedding_model, index_type ("exact"), reranked (false), file_hash, content_hash, citation_id.

Errors

  NestError (format crate): MagicMismatch, UnsupportedVersion, UnsupportedFormatVersion, UnsupportedSchemaVersion, UnsupportedSectionVersion, UnsupportedSectionEncoding, MalformedSectionPayload, FileTruncated, FileSizeMismatch, UnexpectedEof, MissingRequiredSection, SectionOffsetOutOfBounds, SectionMisaligned, SectionNotFound, SectionCountMismatch, InvalidHeaderChecksum, SectionChecksumMismatch, FooterHashMismatch, ManifestInvalid, UnsupportedDType, UnsupportedMetric, UnsupportedScoreType, UnsupportedNormalize, UnsupportedIndexType, UnsupportedRerankPolicy, InvalidModelHash, EmbeddingSizeMismatch, InvalidEmbeddingValue, DimensionMismatch, EmptyQuery, ZeroNormQuery, InvalidK, Json, Io, FileNotFound, InvalidInput, UnsupportedFeature, QueryValidation.

  RuntimeError (runtime crate): wraps NestError plus DimensionMismatch, InvalidK, EmptyQuery, ZeroNormQuery, InvalidQueryValue, Io.

  No panic path is acceptable in library code.

Workspace

  crates/nest-format   standalone library (layout, sections, manifest, reader, writer, chunk, hashing)
  crates/nest-runtime  depends on nest-format (mmap-backed search via MmapNestFile)
  crates/nest-cli      depends on nest-format + nest-runtime (clap binary)
  crates/nest-python   depends on nest-format + nest-runtime (cdylib, PyO3 abi3-py312)
  python/nest.py       dynamic loader, re-exports NestFile, SearchHit, open, build, chunk_id
  python/builder.py    Pipeline class: chunker + SQLite scratch cache + emit + auto-validate
  python/convert_legacy.py  CLI converter from legacy SQLite .nest to v1 binary

Rust API (nest-format)

  pub use chunk::{ChunkInput, chunk_id};
  pub use error::{NestError, Result};
  pub use layout::*;
  pub use manifest::{Capabilities, Manifest};
  pub use reader::NestView;
  pub use sections::{OriginalSpan, SearchContract, decode_chunk_ids, decode_chunks_canonical, decode_chunks_original_spans, decode_provenance, decode_search_contract};
  pub use writer::NestFileBuilder;

  NestFileBuilder::new(manifest: Manifest) -> Self
    .add_chunk(chunk: ChunkInput) -> Self             (consuming builder)
    .add_chunks(chunks: impl IntoIterator<Item = ChunkInput>) -> Self
    .with_provenance(value: serde_json::Value) -> Self
    .reproducible(on: bool) -> Self
    .build_bytes() -> Result<Vec<u8>>
    .write_to_path(path: impl AsRef<Path>) -> Result<()>

  Manifest fields: format_version, schema_version, embedding_model, embedding_dim, n_chunks, dtype, metric, score_type, normalize, index_type, rerank_policy, model_hash, chunker_version, capabilities, title, version, created, description, authors, license, extra (BTreeMap).

  Manifest::validate() checks all v1 contract constraints (version, dtype, metric, score_type, normalize, index_type, rerank_policy, model_hash format, capabilities).

  chunk_id(canonical_text, source_uri, byte_start, byte_end, chunker_version) -> String
    Returns sha256:<64 hex>.

  ChunkInput { canonical_text: String, source_uri: String, byte_start: u64, byte_end: u64, embedding: Vec<f32> }

  NestView::from_bytes(data: &[u8]) -> Result<Self>
    Zero-copy view over a byte slice. Validates magic, header, sections, footer, manifest.
    .get_section_data(section_id: u32) -> Result<&[u8]>
    .validate_embeddings_values() -> Result<()>
    .search_contract() -> Result<SearchContract>
    .file_hash_hex() -> String                          sha256:<hex> of entire file
    .content_hash_hex() -> Result<String>               sha256:<hex> of canonical sections

  Layout constants: NEST_MAGIC, NEST_VERSION_MAJOR (1), NEST_VERSION_MINOR (0), NEST_FORMAT_VERSION (1), NEST_SCHEMA_VERSION (1), NEST_HEADER_SIZE (128), NEST_SECTION_ENTRY_SIZE (32), NEST_FOOTER_SIZE (40), SECTION_ALIGNMENT (64), REPRODUCIBLE_CREATED ("1970-01-01T00:00:00Z"). Section IDs: SECTION_CHUNK_IDS (0x01), SECTION_CHUNKS_CANONICAL (0x02), SECTION_CHUNKS_ORIGINAL_SPANS (0x03), SECTION_EMBEDDINGS (0x04), SECTION_PROVENANCE (0x05), SECTION_SEARCH_CONTRACT (0x06).

Rust API (nest-runtime)

  MmapNestFile::open(path: &Path) -> Result<Self, RuntimeError>
    Opens file, mmaps it, validates everything, indexes embeddings offset.
    .embedding_dim() -> usize
    .n_embeddings() -> usize
    .file_hash() -> &str
    .content_hash() -> &str
    .search(query: &[f32], k: i32) -> Result<SearchResult, RuntimeError>
    .inspect_json() -> Result<String, RuntimeError>
    .revalidate() -> Result<(), RuntimeError>

  SearchHit { chunk_id, score, score_type, source_uri, offset_start, offset_end, embedding_model, index_type, reranked, file_hash, content_hash, citation_id }
  SearchResult { hits: Vec<SearchHit>, query_time_ms, index_type, recall, truncated, k_requested, k_returned }

Python API

  import sys; sys.path.insert(0, "python"); import nest

  nest.open(path) -> NestFile
  NestFile.search(query: list[float], k: int) -> list[SearchHit]
  NestFile.validate() -> bool (raises ValueError on failure)
  NestFile.inspect() -> dict
  NestFile.embedding_dim -> int
  NestFile.n_embeddings -> int
  NestFile.file_hash -> str
  NestFile.content_hash -> str

  SearchHit has attributes: chunk_id, score, score_type, source_uri, offset_start, offset_end, embedding_model, index_type, reranked, file_hash, content_hash, citation_id

  nest.build(output_path, embedding_model, embedding_dim, chunker_version, model_hash, chunks, *, title=None, version=None, created=None, description=None, authors=None, license=None, provenance=None, reproducible=False) -> str

  chunks is a list of dicts with keys: canonical_text, source_uri, byte_start, byte_end, embedding (list[float], length == embedding_dim)

  nest.chunk_id(canonical_text, source_uri, byte_start, byte_end, chunker_version) -> str

  builder.Pipeline(cfg: BuildConfig, embedder: Callable, scratch_db: str | None)
    .add(spec: ChunkSpec) -> None
    .add_many(specs: Iterable[ChunkSpec]) -> None
    .emit(*, provenance: dict | None = None) -> str   (writes file, validates, returns path)
    .close() -> None

  builder.chunk_text(text, source_uri, *, max_chars=512, overlap=0) -> list[ChunkSpec]
  builder.BuildConfig(output_path, embedding_model, embedding_dim, chunker_version, model_hash, ...)
  builder.EmbeddingCache(path) -> get(chunk_id, dim), put(chunk_id, embedding), close()

CLI

  nest inspect <file>              dump header, section table, manifest, hashes
  nest validate <file>              full integrity check (magic, checksums, footer, manifest, contract)
  nest stats <file>                 file size, chunk count, dim, model, per-section sizes
  nest search <file> <query> -k K  exact top-k search; query is JSON array of f32
  nest benchmark <file> -q N -k K  latency stats over N random queries
  nest cite <file> <citation>       resolve nest://<content_hash>/<chunk_id> to canonical text and span

Install and build

  Requirements: Rust (edition 2024), Python 3.12+.

  cargo build --release --workspace
  cp target/release/lib_nest.dylib python/_nest.so    (macOS)
  cp target/release/lib_nest.so python/_nest.so       (linux)

  No maturin. No PyPI package. The extension is a plain cdylib.

  target/release/nest --help

Test

  cargo test --workspace                  all Rust tests
  cargo test -p nest-format               format crate only
  cargo test -p nest-runtime              runtime crate only
  cargo test -p nest-cli                  CLI integration tests (requires release build)

  python tests/test_e2e.py               Python e2e (requires built .so)
  python tests/test_builder.py            builder pipeline + cache (requires built .so)

  No pytest. Tests are plain scripts with if __name__ == "__main__".

  Golden fixture: crates/nest-format/tests/fixtures/golden_v1_minimal.nest (1366 bytes). Regenerate: cargo run -p nest-format --example regen_golden

Lint and format

  cargo fmt --all --check
  cargo clippy --workspace -- -D warnings
  ruff check .          (config in pyproject.toml)
  ruff format .          (config in pyproject.toml)

Version skew policy

  file.format_version > reader.NEST_FORMAT_VERSION  -> UnsupportedFormatVersion
  file.schema_version > reader.NEST_SCHEMA_VERSION  -> UnsupportedSchemaVersion
  header.version_major != 1                         -> UnsupportedVersion
  header.version_minor > reader.NEST_VERSION_MINOR   -> UnsupportedVersion

  Lower or equal versions are accepted.

License


  MIT.