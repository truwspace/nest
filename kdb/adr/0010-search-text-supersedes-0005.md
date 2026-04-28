# ADR 0010: `nest search-text` supersedes ADR 0005

- **Status:** Accepted (supersedes ADR 0005)
- **Date:** 2026-04-28
- **Deciders:** project owner

## Context

ADR 0005 refused to add `nest search file.nest "query string"`
because:

1. It would force the binary to ship an embedding model (80–500 MB).
2. It would pin the binary to a single model.
3. There was no way to verify the runtime model matched the corpus.

Phases 0–3 of the 2026-04-28 push closed (3): ADR 0008 makes
`model_hash` a granular reproducible fingerprint, and the runtime can
now refuse a mismatch with a typed error. (1) and (2) we solve by
shelling out to Python — same toolchain that built the corpus, no
ML deps in the Rust binary.

## Decision

Add `nest search-text <file> "query" -k K`:

- Spawns `python3 python/embed_query.py <model> "query"` (overrideable
  with `--embedder PATH`).
- The embedder loads the model declared in `manifest.embedding_model`
  (overrideable with `--model-path PATH` for offline operation).
- The embedder emits a structured JSON document: `{model_hash,
  fingerprint, embedding_model, embedding_dim, vector}`.
- The CLI runs three checks against the manifest BEFORE running search:
  1. `embedding_model` (name) match.
  2. `embedding_dim` match.
  3. `model_hash` match (the strict gate, see ADR 0008).
- On mismatch, fail with a typed error and a hint pointing at
  `--model-path`.
- After validation, route to the declared `index_type`: `exact` →
  `search()`, `hnsw` → `search_ann()`, `hybrid` → `search_hybrid()`.

`nest search` (the JSON-array variant from ADR 0005) is kept as a
lower-level path for callers that already have a vector and don't
want the Python round-trip. Both paths produce identical hits for the
same query vector.

`--skip-model-hash-check` exists for searching pre-Phase-3 corpora
that still carry the legacy zero-placeholder. The flag's help text
names the risk.

## Consequences

### Positive

- The natural ergonomic (`nest search-text file "query"`) works
  out-of-the-box for users who already have sentence-transformers
  installed.
- Offline operation is a single flag away (`--model-path`). Used
  for fully air-gapped distributions.
- The `nest` Rust binary stays ML-free: search-text is a 5-line
  shell-out, not a bundled model.
- The mismatch failure mode is loud, typed, and reproducible — it
  fails at validation time, not at runtime as silently-bad scores.

### Negative

- New runtime dependency: a `python3` interpreter on `PATH` with
  `sentence_transformers` installed. We document this in
  `doc/usage.md` and degrade gracefully if the embedder is missing.
- Adds an indirection: the CLI spawns a subprocess per query. Fine
  for one-shot CLI use; not appropriate for a high-QPS server. (The
  Python bindings — `nest.NestFile.search()` — bypass this path
  entirely and run in-process.)

### Trade-offs

- Did **not** rewrite the embedder in Rust (candle / onnxruntime).
  The Python toolchain is what built the corpus; using the same
  toolchain at query time guarantees vector parity. A future Rust
  embedder could replace this via the same JSON contract.

## Alternatives considered

- **Bundle a single default model.** Rejected (same as ADR 0005).
- **Auto-detect text vs JSON.** Rejected: silent path-selection
  is the failure mode this ADR exists to avoid.
- **Make `nest search` polymorphic (text OR JSON).** Rejected:
  separate verbs (`search` / `search-text`) make the contract
  explicit and let `--help` document each path's failure modes.

## References

- `crates/nest-cli/src/cmd/search_text.rs` (validation + dispatch).
- `python/embed_query.py` (embedder, structured JSON output).
- `tests/test_search_text_model_hash.py` (5-case E2E gate).
- ADR 0005 (superseded — kept for historical context).
- ADR 0008 (model_hash fingerprint, the gate this ADR depends on).
