# nest

Portable, hash-verified, mmap-friendly container for semantic
knowledge: canonical chunks, their float32 embeddings, original-document
spans, and a runtime contract that pins every search-time decision.

`.nest` v1 is **frozen** — see [`SPEC.md`](SPEC.md) for the byte-by-byte
specification and [`CHANGELOG.md`](CHANGELOG.md) for the release notes.

## Principles

- **Reader-first.** Every read path validates magic, header checksum,
  section checksums, footer hash, manifest contract, required-section
  presence, and embedding values. Failures return typed errors, never
  panics.
- **Recall before throughput.** v1 ships flat exact search only; ANN
  arrives in a future format version.
- **Reproducible.** With `--reproducible`, two builds with identical
  inputs produce byte-identical files.
- **Citable.** Every chunk has a stable `chunk_id`; every hit is
  resolvable via `nest://<content_hash>/<chunk_id>` against the file's
  own hashes.

## Install

Requirements: Rust (edition 2024), Python 3.12+ (3.14 used in dev).

```bash
# Build the workspace (CLI + runtime + format + Python extension)
cargo build --release --workspace

# Install the Python extension into the venv
cp target/release/lib_nest.dylib python/_nest.so   # macOS
# cp target/release/lib_nest.so   python/_nest.so   # Linux

# Sanity-check
target/release/nest --help
python -c "import sys; sys.path.insert(0, 'python'); import nest; print(nest.__name__)"
```

There is no PyPI package and no `maturin develop` requirement — the
extension is a plain `cdylib` copied into `python/`.

## Quickstart — Python build → open → search → cite

```python
import sys
sys.path.insert(0, "python")
import nest

# 1. Build a tiny .nest from in-memory chunks (already embedded).
chunks = [
    dict(
        canonical_text="hello world",
        source_uri="doc.txt",
        byte_start=0,
        byte_end=11,
        embedding=[1.0, 0.0, 0.0, 0.0],   # length must == embedding_dim
    ),
    dict(
        canonical_text="goodbye",
        source_uri="doc.txt",
        byte_start=11,
        byte_end=18,
        embedding=[0.0, 1.0, 0.0, 0.0],
    ),
]

nest.build(
    output_path="demo.nest",
    embedding_model="demo-model",
    embedding_dim=4,
    chunker_version="demo/1",
    model_hash="sha256:" + "0" * 64,
    chunks=chunks,
    reproducible=True,        # bit-for-bit reproducible builds
)

# 2. Open the file via mmap and search.
db = nest.open("demo.nest")
print(db.embedding_dim, db.n_embeddings, db.file_hash, db.content_hash)

# 3. Validate at any time (rerun every reader-side check).
assert db.validate() is True

# 4. Search with a pre-embedded query vector.
hits = db.search([1.0, 0.0, 0.0, 0.0], k=1)
hit = hits[0]
print(hit.score, hit.score_type)            # ~1.0  cosine
print(hit.source_uri, hit.offset_start, hit.offset_end)
print(hit.citation_id)                      # nest://<content_hash>/<chunk_id>
```

## Quickstart — CLI

```bash
# Inspect / validate / stats
target/release/nest inspect  demo.nest
target/release/nest validate demo.nest
target/release/nest stats    demo.nest

# Search. The query is a JSON array of f32 — the CLI does NOT bundle
# an embedding model. For text queries, embed first (see next section).
target/release/nest search demo.nest "[1.0, 0.0, 0.0, 0.0]" --k 5

# Resolve a citation produced by `nest search` back to canonical text.
target/release/nest cite demo.nest \
  "nest://sha256:<content_hash>/sha256:<chunk_id>"
```

`nest cite` returns:

```
citation_id:  nest://<content_hash>/<chunk_id>
file_hash:    sha256:...
content_hash: sha256:...
chunk_id:     sha256:...
source_uri:   <source_uri from chunks_original_spans>
byte_start:   <u64>
byte_end:     <u64>
text:         <canonical text from chunks_canonical>
```

It rejects mismatched `content_hash` with a hard error (exit 1).

## Text queries

The CLI v0.1.0 deliberately accepts only a JSON array of f32 — there is
no embedding model bundled in the binary. To search by text, embed the
query in Python first and pass the JSON to the CLI:

```bash
QVEC=$(python - <<'PY'
from sentence_transformers import SentenceTransformer
import json
m = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")
v = m.encode(["vacina covid"], normalize_embeddings=True)[0].tolist()
print(json.dumps(v))
PY
)
target/release/nest search data/truw_ptbr.v1.nest "$QVEC" --k 5
```

Or stay in Python and skip the shell entirely:

```python
import sys; sys.path.insert(0, "python")
import nest
from sentence_transformers import SentenceTransformer

db    = nest.open("data/truw_ptbr.v1.nest")
model = SentenceTransformer(db.inspect()["manifest"]["embedding_model"])
qvec  = model.encode(["vacina covid"], normalize_embeddings=True)[0].tolist()
for h in db.search(qvec, 5):
    print(f"{h.score:+.4f}  {h.source_uri}  {h.citation_id}")
```

## Repo layout

```
crates/
  nest-format/    on-disk format: layout, sections, manifest, reader, writer
  nest-runtime/   mmap-backed search runtime
  nest-cli/       `nest` binary (inspect/validate/stats/search/benchmark/cite)
  nest-python/    PyO3 bindings (cdylib)
python/
  nest.py         user-facing wrapper (open / search / build / chunk_id)
  builder.py      reusable pipeline: chunker + SQLite scratch + emit + validate
  convert_legacy.py   convert legacy SQLite-based truw_ptbr.nest → v1 binary
tests/
  test_e2e.py     PyO3-only end-to-end
  test_builder.py builder pipeline + cache reuse
SPEC.md           byte-by-byte v1 specification
CHANGELOG.md      release notes
```

## License

MIT. See [`LICENSE`](LICENSE).
