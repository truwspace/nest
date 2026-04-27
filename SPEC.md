# `.nest` v1 — Binary Format Specification

`.nest` v1 is a self-contained, hash-verified, mmap-friendly container for a
semantic knowledge base: canonical chunks, their embeddings, their original
spans, and a runtime contract that pins every search-time decision.

This document is the byte-by-byte source of truth for v1. The Rust
implementation in `crates/nest-format` is the canonical encoder/decoder; this
spec is what every other implementation must agree with.

The format is **frozen** at v1: any change that alters bytes on disk must
bump `NEST_FORMAT_VERSION`. Manifest-only schema additions bump
`NEST_SCHEMA_VERSION`.

---

## 1. Principles

- **Reader-first.** The reader rejects any malformed file with a typed
  error. No "best effort" parsing.
- **Hash everything.** Header, every section, the file as a whole, and a
  `content_hash` over the canonical sections. No silent corruption.
- **Recall before throughput.** v1 ships flat exact search only. ANN
  arrives in a future format version with `index_type` declaring it.
- **Reproducible by construction.** Two builds with identical inputs
  produce byte-identical output when `--reproducible` is set.
- **Citable.** Every chunk has a stable `chunk_id`; every hit is
  resolvable via `nest://<content_hash>/<chunk_id>` against the file's
  own hash.

## 2. Primitives

- **Endianness.** All multi-byte integers are little-endian, unsigned
  unless explicitly marked otherwise. There are no signed integers in
  the v1 wire format.
- **Strings.** Length-prefixed UTF-8: `u32 LE length` then `length`
  bytes. No NUL terminators.
- **Hash.** SHA-256 is the only hash function used. Fields named
  `*_hash` and `*_checksum` are SHA-256 outputs (or truncations thereof,
  always documented). BLAKE3 is not used in v1.
- **Hex.** Hex strings are lowercase. The format `sha256:<64 hex chars>`
  is used wherever a hash is exposed as text.
- **Alignment.** Every section payload starts at a 64-byte aligned
  offset (`SECTION_ALIGNMENT = 64`). The bytes between the previous
  section's end and the next section's start are padding, are filled
  with zero, and are **not** part of any section's checksum.

## 3. File layout

```text
+----------------------------------------------------------------------+
| [0   .. 128)        NestHeader                                        |
| [128 .. 128+T*32)   SectionTable, T = section_table_count entries     |
| [M   .. M+S)        Manifest (JCS canonical JSON, no whitespace)      |
| ... aligned to 64 ...                                                 |
| [O0  .. O0+L0)      section[0] data       \                           |
| ... aligned to 64 ...                       \                         |
| [O1  .. O1+L1)      section[1] data          | each O_i % 64 == 0     |
| ...                                          /                        |
| [On-1.. On-1+Ln-1)  section[n-1] data     /                           |
| [F   .. F+40)       Footer                                            |
+----------------------------------------------------------------------+
F = file_size - 40
```

Layout invariants:

- `section_table_offset == 128`.
- `manifest_offset == 128 + section_table_count * 32`.
- For every entry `i` in the table, `entry.offset % 64 == 0`.
- The first section starts at `align_up_64(manifest_offset + manifest_size)`.
- Section `i+1` starts at `align_up_64(section[i].offset + section[i].size)`.
- The footer starts at the unaligned end of the last section, i.e.
  `last.offset + last.size`.
- `file_size == footer_offset + 40`.

## 4. Header — 128 bytes (`NestHeader`)

| Offset | Size | Field                  | Value / semantics                                  |
| ------ | ---- | ---------------------- | -------------------------------------------------- |
| 0      | 4    | `magic`                | ASCII `"NEST"` (`0x4E 0x45 0x53 0x54`)             |
| 4      | 2    | `version_major`        | `1`                                                |
| 6      | 2    | `version_minor`        | `0`                                                |
| 8      | 4    | `flags`                | reserved, MUST be `0` in v1                        |
| 12     | 4    | `embedding_dim`        | embedding dimensionality, > 0                      |
| 16     | 8    | `n_chunks`             | number of chunks                                   |
| 24     | 8    | `n_embeddings`         | always equal to `n_chunks` in v1                   |
| 32     | 8    | `file_size`            | total file size in bytes (must equal actual size)  |
| 40     | 8    | `section_table_offset` | always `128`                                       |
| 48     | 8    | `section_table_count`  | number of section table entries                    |
| 56     | 8    | `manifest_offset`      | byte offset of the manifest                        |
| 64     | 8    | `manifest_size`        | manifest length in bytes                           |
| 72     | 8    | `header_checksum`      | first 8 bytes of `SHA-256(header[0..72] ++ header[80..128])` |
| 80     | 48   | `reserved`             | MUST be all zero in v1                             |

The header checksum hashes the header with its own checksum slot zeroed
out, so a reader can verify the header in place before trusting any of
its offsets.

## 5. Section table entry — 32 bytes (`SectionEntry`)

| Offset | Size | Field        | Semantics                                            |
| ------ | ---- | ------------ | ---------------------------------------------------- |
| 0      | 4    | `section_id` | one of the section IDs in §6                         |
| 4      | 4    | `encoding`   | `0 = raw`. v1 rejects any other value.               |
| 8      | 8    | `offset`     | byte offset of the payload, MUST satisfy `offset % 64 == 0` |
| 16     | 8    | `size`       | payload length in bytes (excludes padding)           |
| 24     | 8    | `checksum`   | first 8 bytes of `SHA-256(file[offset..offset+size])` |

Reserved encoding values for future versions: `1 = zstd`, `2 = float16`,
`3 = int8`. A v1 reader MUST refuse any value other than `0` with
`UnsupportedSectionEncoding`.

## 6. Required sections (v1)

Every v1 file MUST contain exactly the following six sections, in any
order in the file but with strictly increasing `section_id`s in the
section table (the writer sorts by `section_id`).

| ID     | Name (canonical)         | Payload                                           |
| ------ | ------------------------ | ------------------------------------------------- |
| `0x01` | `chunk_ids`              | length-prefixed UTF-8 strings, count == `n_chunks` |
| `0x02` | `chunks_canonical`       | length-prefixed UTF-8 strings, count == `n_chunks` |
| `0x03` | `chunks_original_spans`  | per-chunk `(source_uri, byte_start, byte_end)`     |
| `0x04` | `embeddings`             | raw `float32 LE`, shape `[n_chunks, embedding_dim]` |
| `0x05` | `provenance`             | versioned JSON blob (free-form, deterministic)     |
| `0x06` | `search_contract`        | versioned JSON document mirroring the manifest     |

The set is closed in v1: a reader MUST reject a file that omits any of
these with `MissingRequiredSection(name)`. Adding a section ID without
bumping `format_version` is forbidden because `content_hash` (§13) is
defined over the canonical alphabetical name list.

### 6.1 Section payload prefix

Every section except `embeddings` (`0x04`) begins with a 12-byte
prefix:

| Offset | Size | Field             | Value          |
| ------ | ---- | ----------------- | -------------- |
| 0      | 4    | `payload_version` | `1` in v1      |
| 4      | 8    | `count` / `length`| section-specific (see below) |

A reader rejects any unknown `payload_version` with
`UnsupportedSectionVersion`.

### 6.2 `chunk_ids` (`0x01`)

```text
prefix.payload_version = 1
prefix.count           = n_chunks
for i in 0..n_chunks:
    u32 LE  len_i
    bytes   chunk_id_i (UTF-8, len_i bytes, "sha256:<64 hex>")
```

`chunk_id_i` is derived deterministically from chunk content (§12).
Every entry MUST start with `"sha256:"` followed by 64 lowercase hex
characters.

### 6.3 `chunks_canonical` (`0x02`)

```text
prefix.payload_version = 1
prefix.count           = n_chunks
for i in 0..n_chunks:
    u32 LE  len_i
    bytes   canonical_text_i (UTF-8)
```

### 6.4 `chunks_original_spans` (`0x03`)

```text
prefix.payload_version = 1
prefix.count           = n_chunks
for i in 0..n_chunks:
    u32 LE  len_i
    bytes   source_uri_i (UTF-8)
    u64 LE  byte_start_i
    u64 LE  byte_end_i        (byte_end >= byte_start)
```

`byte_start` and `byte_end` index into the UTF-8 bytes of the original
source identified by `source_uri_i`. They are NOT character offsets and
are NOT relative to `canonical_text_i`. The relationship between
canonical text and original text (whitespace normalization, casing,
etc.) is the chunker's responsibility — `canonical_text` is what the
embedding was computed over, the span points back to the source for
provenance.

### 6.5 `embeddings` (`0x04`)

No prefix. The payload is exactly `n_embeddings * embedding_dim * 4`
bytes of `float32 LE`. Element ordering matches `chunk_ids` and
`chunks_canonical`: `embeddings[i]` is the embedding for `chunk_id[i]`.
Vectors MUST be L2-normalized (the manifest declares `normalize = "l2"`)
and MUST contain neither `NaN` nor `Inf`.

### 6.6 `provenance` (`0x05`)

```text
u32 LE  payload_version = 1
u64 LE  json_len
bytes   json_payload (UTF-8 JSON)
```

The JSON is free-form so the format does not constrain ingestion
metadata, but it MUST round-trip through `serde_json` (so it must be
valid JSON). Reasonable conventions:

- `legacy_source` / `legacy_version` — origin description.
- `retrieved_at` — ISO-8601 timestamp the source was crawled.
- `license` — SPDX identifier or free text (the manifest also carries a
  `license` field for the dataset as a whole; per-source licenses go
  here).
- `labels`, `sources` — per-chunk auxiliary metadata.

### 6.7 `search_contract` (`0x06`)

```text
u32 LE  payload_version = 1
u64 LE  json_len
bytes   json_payload (UTF-8 JSON)
```

The JSON MUST conform to:

```json
{
  "metric":         "ip",
  "score_type":     "cosine",
  "normalize":      "l2",
  "index_type":     "exact",
  "rerank_policy":  "none"
}
```

The reader cross-checks this with the manifest at open time; any
disagreement is rejected with the relevant `Unsupported*` error.

## 7. Footer — 40 bytes (`NestFooter`)

| Offset | Size | Field        | Semantics                                       |
| ------ | ---- | ------------ | ----------------------------------------------- |
| 0      | 8    | `footer_size`| `40`                                             |
| 8      | 32   | `file_hash`  | `SHA-256(file[0..file_size-40])` (32 bytes)      |

`file_hash` covers everything before the footer, **including** padding.
A bit-flip anywhere in the file (header, section table, manifest, any
section payload, or any padding byte) produces a footer hash mismatch.

## 8. Hashing summary

| Name              | Algorithm | Width | Coverage                                       |
| ----------------- | --------- | ----- | ---------------------------------------------- |
| `header_checksum` | SHA-256   | 8 B   | header bytes `[0..72) ++ [80..128)`            |
| section `checksum`| SHA-256   | 8 B   | `file[offset..offset+size]`, padding excluded  |
| `file_hash` (footer) | SHA-256 | 32 B | `file[0..file_size-40)`, padding included      |
| `content_hash`    | SHA-256   | 32 B  | canonical sections in alphabetical order (§13) |
| `chunk_id`        | SHA-256   | 32 B  | domain-separated preimage (§12)                |
| `model_hash`      | SHA-256   | 32 B  | caller-supplied; format `sha256:<64 hex>`      |

## 9. Manifest

The manifest is a single JCS-style canonical JSON document encoded in
UTF-8 with no whitespace. Field order follows declaration order; the
free-form `extra` map is serialized in `BTreeMap` order so it is
deterministic.

| Field                  | Type                        | Required | v1 value / constraint                              |
| ---------------------- | --------------------------- | :------: | -------------------------------------------------- |
| `format_version`       | `u32`                       |  yes     | `1`                                                |
| `schema_version`       | `u32`                       |  yes     | `1`                                                |
| `embedding_model`      | `string`                    |  yes     | non-empty                                          |
| `embedding_dim`        | `u32`                       |  yes     | `> 0`, equal to `header.embedding_dim`             |
| `n_chunks`             | `u64`                       |  yes     | `> 0`, equal to `header.n_chunks`                  |
| `dtype`                | `string`                    |  yes     | `"float32"`                                        |
| `metric`               | `string`                    |  yes     | `"ip"`                                             |
| `score_type`           | `string`                    |  yes     | `"cosine"`                                         |
| `normalize`            | `string`                    |  yes     | `"l2"`                                             |
| `index_type`           | `string`                    |  yes     | `"exact"`                                          |
| `rerank_policy`        | `string`                    |  yes     | `"none"`                                           |
| `model_hash`           | `string`                    |  yes     | `"sha256:" + 64 hex`                               |
| `chunker_version`      | `string`                    |  yes     | non-empty                                          |
| `capabilities`         | object (see below)          |  yes     | `supports_exact = true`, `supports_reproducible_build = true` |
| `title`                | `string`                    |   no     | dataset display name                               |
| `version`              | `string`                    |   no     | dataset version (caller-defined)                   |
| `created`              | `string` (ISO-8601 UTC)     |   no     | `"1970-01-01T00:00:00Z"` in `--reproducible` mode  |
| `description`          | `string`                    |   no     |                                                    |
| `authors`              | `array<string>`             |   no     |                                                    |
| `license`              | `string`                    |   no     | SPDX identifier                                    |
| (`extra`)              | `map<string, json>`         |   no     | reader-tolerant additions; serialized BTreeMap     |

`capabilities` object:

```json
{
  "supports_exact":              true,
  "supports_ann":                false,
  "supports_bm25":               false,
  "supports_citations":          true,
  "supports_reproducible_build": true
}
```

`supports_exact` and `supports_reproducible_build` MUST be `true` in v1;
the others are advisory forward-looking flags so a runtime can short-
circuit without scanning sections.

## 10. Reproducibility

When the writer is invoked with the reproducible flag (Rust:
`NestFileBuilder::reproducible(true)`; Python: `nest.build(...,
reproducible=True)`), the writer overrides `manifest.created` to the
constant string `"1970-01-01T00:00:00Z"` (`REPRODUCIBLE_CREATED`).

Two builds with identical inputs (chunks, embeddings, manifest fields
other than `created`, provenance JSON) MUST produce byte-identical
files. This is enforced by:

- `crates/nest-format/src/writer.rs::reproducible_mode_overrides_created`
- `tests/test_e2e.py::test_reproducible_builds_match_byte_for_byte`

The provenance JSON is NOT rewritten by reproducible mode — callers are
responsible for keeping provenance deterministic if they want
bit-for-bit equality. The legacy converter at `python/convert_legacy.py`
demonstrates this.

## 11. Version skew

A reader rejects a file whose `format_version` or `schema_version` is
strictly greater than the reader's own `NEST_FORMAT_VERSION` /
`NEST_SCHEMA_VERSION`. A reader accepts files with equal or smaller
versions, on the assumption that earlier versions used a strict subset
of the current contract.

```text
file.format_version > reader.NEST_FORMAT_VERSION  → UnsupportedFormatVersion
file.schema_version > reader.NEST_SCHEMA_VERSION  → UnsupportedSchemaVersion
header.version_major != 1                         → UnsupportedVersion
header.version_minor > reader.NEST_VERSION_MINOR  → UnsupportedVersion
```

## 12. `chunk_id` derivation

The `chunk_id` is a domain-separated SHA-256 over the canonical text,
its original span, and the chunker version. The preimage layout is
fixed by spec — anything else lets two runs of the same chunker produce
different IDs for the same content, which breaks reproducibility.

```text
preimage =
    "nest:chunk_id:v1\n"                  (17 bytes ASCII, includes the newline)
    u32 LE  len(canonical_text)
    bytes   canonical_text
    u32 LE  len(source_uri)
    bytes   source_uri
    u64 LE  byte_start
    u64 LE  byte_end
    u32 LE  len(chunker_version)
    bytes   chunker_version

chunk_id = "sha256:" + lower_hex(SHA-256(preimage))
```

## 13. `content_hash`

`content_hash` is a stable identifier for the *content* of a `.nest`
file, independent of the file-level metadata that may change without
the data changing (e.g. the `created` timestamp). It is the input to
the citation URI (§14).

```text
ordered = sort canonical-section-name lexicographically:
          [chunk_ids, chunks_canonical, chunks_original_spans,
           embeddings, provenance, search_contract]

h = SHA-256()
for (id, name) in ordered:
    h.update(u32 LE len(name))
    h.update(name as UTF-8)
    h.update(u64 LE size(section_data))
    h.update(section_data)

content_hash = "sha256:" + lower_hex(h.finalize())
```

The order is fixed by spec (sorted alphabetically) so future section
additions cannot reshuffle existing files' content_hash. Padding is
NOT included.

## 14. Citation URI

```text
nest://<content_hash>/<chunk_id>
```

Where `content_hash` and `chunk_id` are full `sha256:<hex>` strings
(NOT just the hex tails). Resolution (`nest cite`) requires:

- `content_hash` matches `view.content_hash_hex()` of the target file.
- `chunk_id` is present in the file's `chunk_ids` section.

A successful resolve returns `(canonical_text, source_uri, byte_start,
byte_end)`. Citations are stable across builds of the same content even
if the wrapper file is rebuilt with different `created`/metadata.

## 15. Errors

The reader emits a typed `NestError` per failure mode. The runtime
wraps these in `RuntimeError`. Categories:

| Category   | Variants                                                        |
| ---------- | --------------------------------------------------------------- |
| Format     | `MagicMismatch`, `UnsupportedVersion`, `UnsupportedFormatVersion`, `UnsupportedSchemaVersion`, `UnsupportedSectionVersion`, `UnsupportedSectionEncoding`, `MalformedSectionPayload`, `FileTruncated`, `FileSizeMismatch`, `UnexpectedEof`, `MissingRequiredSection`, `SectionOffsetOutOfBounds`, `SectionMisaligned`, `SectionNotFound`, `SectionCountMismatch` |
| Hash       | `InvalidHeaderChecksum`, `SectionChecksumMismatch`, `FooterHashMismatch` |
| Manifest   | `ManifestInvalid`, `UnsupportedDType`, `UnsupportedMetric`, `UnsupportedScoreType`, `UnsupportedNormalize`, `UnsupportedIndexType`, `UnsupportedRerankPolicy`, `InvalidModelHash` |
| Embedding  | `EmbeddingSizeMismatch`, `InvalidEmbeddingValue`               |
| Query      | `EmptyQuery`, `ZeroNormQuery`, `InvalidEmbeddingValue` (in query), `DimensionMismatch`, `QueryValidation` |
| Bookkeeping| `InvalidK`, `Io`, `Json`, `FileNotFound`, `InvalidInput`, `UnsupportedFeature` |

No panic path is acceptable. A reader that hits an unexpected condition
MUST return a `NestError`.

## 16. Search contract

v1 ships exactly one search mode: flat exact. The runtime guarantees:

- Query `[f32; embedding_dim]` is L2-normalized internally (the file's
  embeddings are already normalized, so the resulting score is
  cosine ∈ `[-1, 1]`).
- Score is the real cosine value, not a transformed proxy.
- Top-`k` is a stable sort by score descending, breaking ties by index
  ascending.
- `recall == 1.0` always for exact mode.
- `truncated == true` iff `k < n_embeddings`.
- `k_returned == min(k, n_embeddings)`.
- Validations: `k > 0`, query non-empty, `query.len() == embedding_dim`,
  no `NaN`/`Inf` in query, query L2 norm > 0.

Each `SearchHit` carries:

```text
chunk_id        (sha256:<hex>)
score           (f32, real cosine)
score_type      ("cosine")
source_uri      (from chunks_original_spans)
offset_start    (byte_start)
offset_end      (byte_end)
embedding_model (from manifest)
index_type      ("exact")
reranked        (false in v1)
file_hash       (sha256:<hex>, the file's footer hash)
content_hash    (sha256:<hex>, see §13)
citation_id     ("nest://" + content_hash + "/" + chunk_id)
```

## 17. CLI surface

| Command              | Behavior                                                       |
| -------------------- | -------------------------------------------------------------- |
| `nest inspect <f>`   | dump header, section table, manifest, hashes                   |
| `nest validate <f>`  | full integrity check (header / sections / footer / manifest)   |
| `nest stats <f>`     | size, chunk count, dim, model, hashes, per-section sizes       |
| `nest search <f> Q -k K` | exact top-k search; Q is a JSON array of f32                |
| `nest benchmark <f> -q N -k K` | latency stats over N random queries                  |
| `nest cite <f> URI`  | resolve `nest://...` into `(text, span, hashes)`               |

## 18. Python bindings

Built into `python/_nest.so` (PyO3, abi3-py312). High-level wrapper at
`python/nest.py` re-exports:

| Python symbol                | Backing                                                     |
| ---------------------------- | ----------------------------------------------------------- |
| `nest.open(path) -> NestFile`| `MmapNestFile::open`                                        |
| `NestFile.search(q, k)`      | exact flat search                                           |
| `NestFile.inspect()`         | dict of header / sections / manifest / hashes               |
| `NestFile.validate()`        | re-runs reader validation; raises on any failure            |
| `NestFile.embedding_dim`     | header field                                                |
| `NestFile.n_embeddings`      | header field                                                |
| `NestFile.file_hash`         | footer hash (`sha256:<hex>`)                                |
| `NestFile.content_hash`      | content hash (§13)                                          |
| `nest.build(...)`            | `NestFileBuilder` glue (kwargs map to manifest fields)      |
| `nest.chunk_id(...)`         | `chunk_id` derivation (§12) for dedup                       |

## 19. Reference artefacts

- Golden minimal fixture: `crates/nest-format/tests/fixtures/golden_v1_minimal.nest`
  (1366 bytes; `file_hash`, `content_hash`, `chunk_id`, length asserted in
  `crates/nest-format/tests/golden.rs`).
- Regenerator: `cargo run -p nest-format --example regen_golden` rewrites
  the fixture and prints the new constants.
- Reproducibility test: `tests/test_e2e.py::test_reproducible_builds_match_byte_for_byte`.
- Legacy SQLite dataset: `data/truw_ptbr.nest` (28 MB) →
  `data/truw_ptbr.v1.nest` (73.73 MB, 19,769 chunks, dim 384) via
  `python/convert_legacy.py`.
- Pipeline reference: `python/builder.py` (chunker + SQLite scratch +
  emit + auto-validate).
