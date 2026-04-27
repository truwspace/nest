# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.0] — 2026-04-27

First public release. The on-disk format, hash semantics, citation URI,
manifest contract, and CLI surface listed below are **frozen** for v1: any
change must bump `NEST_FORMAT_VERSION` (binary container) or
`NEST_SCHEMA_VERSION` (manifest fields).

### Frozen binary container

- File magic: `NEST` (`0x4E 0x45 0x53 0x54`).
- `NEST_VERSION_MAJOR = 1`, `NEST_VERSION_MINOR = 0`,
  `NEST_FORMAT_VERSION = 1`, `NEST_SCHEMA_VERSION = 1`.
- Header: 128 bytes, `repr(C)`, compile-time asserted.
  Fields (LE, unsigned): `magic`, `version_major/minor`, `flags`,
  `embedding_dim`, `n_chunks`, `n_embeddings`, `file_size`,
  `section_table_offset`, `section_table_count`, `manifest_offset`,
  `manifest_size`, `header_checksum[8]`, `reserved[48]`.
- Section table entry: 32 bytes, `repr(C)`, compile-time asserted.
  Fields: `section_id(u32)`, `encoding(u32)`, `offset(u64)`,
  `size(u64)`, `checksum[8]`.
- Footer: 40 bytes (`u64 footer_size = 40`, `[u8; 32] file_hash`).
- Section payload alignment: every section's `offset` is a multiple of
  `SECTION_ALIGNMENT = 64`. Padding between sections is zero and is
  excluded from each section's checksum (but covered by the footer
  hash).
- Endianness: little-endian, unsigned unless explicitly noted.

### Required sections (canonical, alphabetical for `content_hash`)

| ID     | Name                       |
| ------ | -------------------------- |
| `0x01` | `chunk_ids`                |
| `0x02` | `chunks_canonical`         |
| `0x03` | `chunks_original_spans`    |
| `0x04` | `embeddings`               |
| `0x05` | `provenance`               |
| `0x06` | `search_contract`          |

Encoding: only `SECTION_ENCODING_RAW = 0` is accepted in v1. Values
`1 = zstd`, `2 = float16`, `3 = int8` are reserved for future versions.

### Hashing

- Primary hash: SHA-256 throughout. No BLAKE3.
- `header_checksum`: first 8 bytes of `SHA-256(header[0..72] ++
  header[80..128])` — header with its own checksum slot zeroed.
- Section `checksum`: first 8 bytes of `SHA-256(payload)`. Padding is
  not hashed.
- `file_hash` (footer): full 32-byte `SHA-256(file[0..file_size-40))`,
  including padding.
- `content_hash`: 32-byte `SHA-256` over the canonical sections in
  alphabetical-by-name order, each domain-separated by length-prefixed
  name and length-prefixed payload. Stable across rebuilds of the same
  content.
- `chunk_id`: domain-separated `SHA-256` with literal preimage prefix
  `"nest:chunk_id:v1\n"`. Format `sha256:<64 hex chars>`.
- `model_hash`: caller-supplied; format `sha256:<64 hex chars>` enforced
  at write time.

### Manifest contract

JCS-style canonical JSON (declaration-ordered known fields, BTreeMap
order for `extra`, no whitespace). Required values for v1:

- `dtype = "float32"`, `metric = "ip"`, `score_type = "cosine"`,
  `normalize = "l2"`, `index_type = "exact"`, `rerank_policy = "none"`.
- `capabilities.supports_exact = true`,
  `capabilities.supports_reproducible_build = true`.
- `model_hash` ∈ `sha256:<64 hex>` regex.

### Reproducibility

- `NestFileBuilder::reproducible(true)` overrides `manifest.created` to
  `"1970-01-01T00:00:00Z"` (`REPRODUCIBLE_CREATED`).
- Two builds with identical inputs produce byte-identical files.
  Verified on the legacy converter (`data/truw_ptbr.nest` → 73.73 MB
  v1 binary); both builds shasum to
  `b9f6e0ea16176706f08767559927737ce91070147ec6cb54e26710bff3d2566d`.

### Version skew policy

- Reader rejects `format_version` or `schema_version` greater than its
  own constants (`NestError::UnsupportedFormatVersion` /
  `UnsupportedSchemaVersion`).
- Reader accepts equal or smaller versions.
- Header version: `version_major != 1` rejected; `version_minor` may
  drift downward.

### CLI v1 surface

Frozen subcommands in `nest-cli`:

| Command         | Behavior                                                       |
| --------------- | -------------------------------------------------------------- |
| `nest inspect`  | header, section table, manifest, hashes                        |
| `nest validate` | full integrity check (header / sections / footer / manifest)   |
| `nest stats`    | size, chunk count, dim, model, hashes, per-section sizes       |
| `nest search`   | exact top-k search; query is a JSON array of f32                |
| `nest benchmark`| latency stats over N random queries                            |
| `nest cite`     | resolve `nest://content_hash/chunk_id` → `(text, span, hashes)` |

The CLI does **not** ship an embedding model. Text → vector is the
caller's responsibility (see `python/builder.py` and
`python/convert_legacy.py` for examples using sentence-transformers).

### Citation URI

`nest://<content_hash>/<chunk_id>` where both halves are full
`sha256:<hex>` strings. `nest cite` rejects citations whose
`content_hash` does not match the file's `content_hash`.

### Search contract

`SearchHit` exposes: `chunk_id`, `score` (real f32 cosine in `[-1, 1]`),
`score_type = "cosine"`, `source_uri`, `offset_start`, `offset_end`,
`embedding_model`, `index_type = "exact"`, `reranked = false`,
`file_hash`, `content_hash`, `citation_id`. Top-k uses a stable sort by
score descending, breaking ties by index ascending.
`recall = 1.0` always; `truncated = (k < n_embeddings)`.

### Test surface

- 70 Rust tests (`cargo test --workspace`):
  - 34 `nest-format` unit tests (layout, manifest, sections, chunk,
    writer)
  - 5 `nest-format` golden-fixture tests (1366-byte minimal `.nest`)
  - 19 `nest-format` roundtrip / negative tests (truncation, magic,
    encoding, alignment, version skew, dim mismatch, NaN/Inf)
  - 8 `nest-runtime` flat-search tests
  - 4 `nest-cli` integration tests (`validate`, `search`, `inspect`,
    `cite`)
- 4 Python tests (`tests/test_e2e.py`, PyO3-only) and 3 builder tests
  (`tests/test_builder.py`).
- `cargo fmt --all --check` clean; `cargo clippy --workspace --
  -D warnings` clean.

### Python bindings

`python/_nest.so` (PyO3, abi3-py312). Wrapper `python/nest.py` exposes:

- `nest.open(path) -> NestFile`
- `NestFile.search(qvec, k) -> [SearchHit]`
- `NestFile.inspect()` / `NestFile.validate()`
- `NestFile.embedding_dim` / `n_embeddings` / `file_hash` / `content_hash`
- `nest.build(...)` (writer glue)
- `nest.chunk_id(...)` (deterministic id derivation)

Single Python entry point: no subprocess CLI fallback inside
`python/`.

### Reference artefacts

- Golden fixture: `crates/nest-format/tests/fixtures/golden_v1_minimal.nest`
  (1366 bytes; regenerate with
  `cargo run -p nest-format --example regen_golden`).
- Legacy SQLite-based dataset: `data/truw_ptbr.nest` (28 MB) →
  `data/truw_ptbr.v1.nest` (73.73 MB, 19,769 chunks dim 384) via
  `python/convert_legacy.py`.
- Specification: `SPEC.md`.

[0.1.0]: https://github.com/truw/nest/releases/tag/v0.1.0
