# ADR 0008: `model_hash` is a granular fingerprint, not a free-form sha256

- **Status:** Accepted
- **Date:** 2026-04-28
- **Deciders:** project owner

## Context

The manifest's `model_hash` is the only field that lets a runtime
verify "the embedder I just ran really is the model that produced the
corpus". `nest search-text` (introduced in this session, see ADR 0010)
needs that gate — without it, a query embedded by a *different*
model with the same name and the same dim returns cosine-valid
garbage with no error surfaced.

The v0.1 contract just required `model_hash` to match the regex
`sha256:[0-9a-f]{64}`. That's a syntactic check; it doesn't say *what*
those 64 hex chars represent. Real-world habit: callers stamped a
zero placeholder (`sha256:` + 64 zeros) and moved on.

A naive fix — "sha256 the model dir" — is unstable. The dir contains
caches, lock files, locale-specific symlinks, and snapshot pointers
that vary across machines while leaving the model's behavior unchanged.

## Decision

`model_hash` is `sha256(canonical_json(fingerprint))` where
`fingerprint` is a structured dict with these fields:

```
model_id              str   — the HF identifier or local path the caller used
files_hash            str   — sha256 over the inference-relevant files only
tokenizer_hash        str   — sha256 of tokenizer.json (if present)
pooling_config_hash   str   — sha256 of 1_Pooling/config.json (if present)
embedding_dim         int
normalize_embeddings  bool
```

`files_hash` only walks a fixed list of inference-affecting files
(see `python/model_fingerprint.py::RELEVANT_FILES`):
`config.json`, `config_sentence_transformers.json`, `modules.json`,
`sentence_bert_config.json`, `tokenizer.json`, `tokenizer_config.json`,
`special_tokens_map.json`, `1_Pooling/config.json`, `model.safetensors`,
`pytorch_model.bin`. Anything else in the snapshot dir is ignored.

The canonical JSON serializer uses `sort_keys=True, separators=(",",
":")` so two callers on different machines that have the same logical
fingerprint get the same `model_hash`.

Two corpora built from the same upstream snapshot, on different
machines, by different humans, get the same `model_hash`. Two
corpora built from snapshots that differ in tokenizer or weights get
different hashes.

`nest search-text` validates in three layers:
1. `manifest.embedding_model` matches the embedder's reported name.
2. `manifest.embedding_dim` matches `len(vector)`.
3. `manifest.model_hash` matches the embedder's recomputed hash.

Layer 3 is the only one that catches "same name, same dim, different
snapshot". Mismatch fails with a typed error and a hint pointing at
`--model-path`.

Pre-Phase-3 corpora (with the legacy zero placeholder) are explicitly
flagged: the runtime refuses to use them via `search-text` unless the
caller passes `--skip-model-hash-check` and acknowledges the risk.

## Consequences

### Positive

- The contract becomes machine-verifiable, not a hint. The runtime
  refuses to silently feed mismatched embeddings into cosine math.
- Two builds of the same corpus on different machines produce the
  same `model_hash` because the fingerprint is reproducible by
  construction.
- `--model-path` becomes useful: a fully offline operator can copy
  the model snapshot dir, the `.nest`, and never touch the
  HuggingFace Hub again. Verification still works.

### Negative

- Adds a Python build-time dependency: callers must compute the
  fingerprint before emitting the manifest. `compute_model_fingerprint`
  is ~80 lines of pure I/O — small but not zero.
- Existing corpora with the placeholder are forward-compat-broken
  for `search-text`. Mitigation: explicit opt-out flag, recommended
  rebuild path documented in ADR text and in `doc/usage.md`.

### Trade-offs

- Did **not** include the model_dir's *path* in the fingerprint —
  paths are machine-local. Did **not** include the snapshot's HF
  revision SHA either: not all snapshots ship with that field, and
  the file_hash already encodes any byte-level difference.

## Alternatives considered

- **Hash the entire model directory.** Rejected: cache/lock/symlink
  contents differ across machines, breaking reproducibility.
- **Trust `embedding_model` (name only).** Rejected: same name +
  same dim across two snapshots is the silent-failure case this ADR
  exists to prevent.
- **Compute the fingerprint inside the Rust runtime.** Rejected:
  loading a sentence-transformers snapshot requires Python anyway,
  and duplicating the file walker in Rust would split the source of
  truth for "what files matter".

## References

- `python/model_fingerprint.py` (ModelFingerprint, compute, resolve_model_dir).
- `python/embed_query.py` (emits structured JSON with the hash).
- `crates/nest-cli/src/cmd/search_text.rs` (three-layer validation).
- `tests/test_search_text_model_hash.py` (5-case E2E gate).
- ADR 0010 (search-text command, the consumer of this gate).
