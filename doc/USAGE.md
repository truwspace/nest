# Usage

`nest` is a single-file binary container for distributing semantic
knowledge bases. One file: chunks, canonical text, byte-spans,
embeddings, search contract, hashes. Copy it, share it, search it.

This guide covers the eight commands you'll actually use.

## 1. Build a `.nest` from chunks

The Python pipeline owns chunking, embedding, caching, and the
final emit. The Rust writer owns reproducibility, hashing, and
deterministic byte layout.

```python
import sys; sys.path.insert(0, "python")
from builder import BuildConfig, ChunkSpec, Pipeline, chunk_text

def embed(specs):
    # plug in your sentence-transformers or candle / onnxruntime here
    from sentence_transformers import SentenceTransformer
    m = SentenceTransformer(cfg.embedding_model)
    return m.encode([s.canonical_text for s in specs],
                    normalize_embeddings=True).tolist()

cfg = BuildConfig(
    output_path="my_corpus.nest",
    embedding_model="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
    embedding_dim=384,
    chunker_version="my-chunker/v1",
    model_hash="sha256:" + "0" * 64,    # see §7 for the real fingerprint
    preset="exact",                      # see §6 for preset choices
    reproducible=True,
)
pipe = Pipeline(cfg, embedder=embed, scratch_db="cache.sqlite")
for source_uri, text in documents:
    for spec in chunk_text(text, source_uri):
        pipe.add(spec)
pipe.emit()
```

For real-world examples: `python/convert_legacy.py` (SQLite → `.nest`) and
`python/tools/nest_build_corpus.py` (7 PT-BR datasets → unified `.nest`).

Direct API (no chunker): `nest.build(output_path, embedding_model, embedding_dim,
chunker_version, model_hash, chunks, preset="exact", reproducible=True)`.

## 2. Validate

Full integrity check: magic, header checksum, every section's
SHA-256 (over physical bytes), footer hash (over the whole file),
manifest schema, contract cross-check against the manifest, NaN/Inf
walk over the embeddings.

```sh
nest validate my_corpus.nest
```

Failure modes are typed (`SectionChecksumMismatch(0x04)`,
`UnsupportedDType("bfloat16")`, etc.) — never "best effort".

## 3. Stats

Sizes, dim, dtype, model, hashes, per-section bytes, the SIMD backend
the runtime selected.

```sh
nest stats my_corpus.nest
```

## 4. Inspect

Header bytes, full section table, manifest as JSON. Use `--json` for
programmatic consumers (CI dashboards, drift detection):

```sh
nest inspect my_corpus.nest             # human-readable
nest inspect my_corpus.nest --json | jq # structured
```

Schema: `{magic, version_major, version_minor, format_version,
schema_version, embedding_dim, n_chunks, n_embeddings, file_size,
manifest, sections[], file_hash, content_hash, simd_backend}`.

## 5. Search

### Exact path (vector input)

Pass a query vector directly as a JSON array. recall = 1.0 by
construction.

```sh
nest search my_corpus.nest "[0.1, 0.2, ...]" -k 10
```

### Search-by-text

Embed the query with the same model the corpus was built with (the
manifest declares it), then route to the declared `index_type`
(exact / hnsw / hybrid). The runtime cross-checks the embedder's
`model_hash` against the manifest before running search and refuses
on mismatch — see §7.

```sh
nest search-text my_corpus.nest "vacina contra covid funciona" -k 5
```

For tuning the candidate set: `--candidates N` (default `4*k`, min 64).

### Force the ANN path

Useful for debugging or measuring `ef_search` curves. Falls back to
exact if the file has no HNSW section.

```sh
nest search-ann my_corpus.nest "[0.1, 0.2, ...]" -k 10 --ef 200
```

## 6. Presets

`preset=` selects a (text encoding, embedding dtype, optional ANN, optional BM25)
bundle. Per-knob overrides win — see `BuildConfig.text_encoding`,
`.dtype`, `.with_hnsw`, `.with_bm25`.

| preset | text | embeddings | ANN | BM25 | size_ratio | recall@10 |
|---|---|---|---|---|---:|---:|
| `exact` | raw | float32 | — | — | 1.000 | 1.0000 |
| `compressed` | zstd | float16 | — | — | 0.350 | 1.0000 |
| `tiny` | zstd | int8 | yes | — | 0.283 | ≥0.95 |
| `hybrid` | zstd | float32 | yes | yes | 0.668 | 1.0000 |

Numbers measured on the project's PT-BR fake-news corpus (n=30725, dim=384).
Latency ranges (NEON, hot cache): exact p50 ≈ 3.0 ms, tiny p50 ≈ 1.3 ms,
hybrid p50 ≈ 4.5 ms.

Pick `tiny` for the smallest distributable file (≈ 30% of `exact`),
`compressed` when you need lossless cosine + 3× compression, `hybrid`
when queries include rare terms / proper nouns / siglas that pure
embeddings underweight, and `exact` when storage isn't the bottleneck
and you want the recall=1.0 ground truth.

## 7. `model_hash` and offline operation (`--model-path`)

`search-text` cross-checks three things before running search:
1. `manifest.embedding_model` (name) matches the embedder's report.
2. `manifest.embedding_dim` matches `len(vector)`.
3. `manifest.model_hash` matches the embedder's reproducible
   fingerprint.

Layer 3 is the only one that catches the silent failure mode "same
name + same dim + different snapshot → cosine-valid garbage". The
fingerprint hashes a fixed list of inference-relevant files
(`config.json`, `tokenizer.json`, `model.safetensors`,
`1_Pooling/config.json`, etc.); see `python/model_fingerprint.py`.

Build with a real fingerprint:

```python
from model_fingerprint import (
    compute_model_fingerprint, fingerprint_to_model_hash, resolve_model_dir,
)
md = resolve_model_dir("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
fp = compute_model_fingerprint(md, model_id="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
cfg.model_hash = fingerprint_to_model_hash(fp)
```

### Fully-offline search

Distribute the model directory alongside the `.nest` (e.g. on a USB
stick or in a sealed Docker image), then point `--model-path` at it
on every search:

```sh
nest search-text my_corpus.nest "vacina contra covid" -k 5 \
    --model-path /mnt/models/paraphrase-multilingual-MiniLM-L12-v2
```

No HuggingFace cache hits, no network. The fingerprint is recomputed
locally and verified against the manifest.

### Pre-Phase-3 corpora

Files built with `model_hash = sha256:0…0` (the legacy placeholder)
fail the strict gate by design. Two options:
- Rebuild with a real fingerprint (recommended).
- Pass `--skip-model-hash-check` to proceed at your own risk. The
  search is still cosine-valid IF you genuinely use the same
  embedding model — but there is no guarantee.

## 8. Benchmark

Random-query latency stats (mean, p50, p95, p99). With `--ann`, also
runs ANN against the same queries and computes `recall@k (ANN vs exact)`.
With `--madvise-cold`, runs an extra pass calling
`posix_madvise(MADV_DONTNEED)` between queries — an upper bound on
cold-cache latency, not absolute cold (see `MmapNestFile::madvise_cold`
docs).

```sh
nest benchmark my_corpus.nest -q 100 -k 10 --ann 100 --madvise-cold
```

Typical output (n=30725, dim=384, neon, int8):
```
Exact (100 queries, dim=384, dtype=int8, simd=neon) [hot]:
  p50: 1.28 ms  p95: 1.68 ms
Exact ... [madvise-cold]:
  p50: 1.95 ms  p95: 2.40 ms
ANN ef=100 (100 queries) [hot]:
  p50: 0.44 ms  p95: 0.62 ms
  recall@10 (ANN vs exact): 0.9920
```

## 9. Citations

Every search hit carries a stable `citation_id` of the form
`nest://<content_hash>/<chunk_id>`. Resolve it back to the canonical
text and original byte span:

```sh
nest cite my_corpus.nest 'nest://sha256:1aa9.../sha256:8f314...'
```

`content_hash` is hashed over the **decoded** bytes, so a corpus
stored with `text_encoding=zstd` produces the same `content_hash` as
the same logical content stored raw — citations are stable across
wire encodings.

## 10. Release verification

```sh
./scripts/release_check.sh
```

Runs the full pipeline: cargo test/clippy/fmt, all 3 Python test
suites, ruff, `measure_presets.py`, `compare_measure.py` against the
committed baseline. Exits non-zero on any failure.
