nest

portable binary format for distributing semantic knowledge bases. one file carries chunks, embeddings, source spans and a search contract, all hash-verified and memory-mapped.

built by truw to run 100 percent local, no server, with cryptographic integrity and traceable citation.

python builds. rust runs. nest packages. agents consume.

install

requires rust edition 2024 and python 3.12+.

```
cargo build --release --workspace
cp target/release/lib_nest.dylib python/_nest.so   # macOS
cp target/release/lib_nest.so python/_nest.so      # linux
```

CLI

```
nest inspect <file>              header, manifest, hashes
nest validate <file>             full integrity check
nest stats <file>                sizes, counts, model
nest search <file> <qvec> -k K   exact top-k, query is a JSON array of f32
nest cite <file> <citation>      resolves nest://<content_hash>/<chunk_id>
```

search takes a vector, not text. embedding text stays in python. deliberate v1 decision.

python

```
import sys; sys.path.insert(0, "python"); import nest

db = nest.open(path)
hits = db.search(qvec, k=5)
hits[0].citation_id   # nest://content_hash/chunk_id
hits[0].source_uri
hits[0].offset_start, hits[0].offset_end
hits[0].score         # real cosine

db.validate()
db.inspect()
```

build a file:

```
nest.build(
    output_path,
    embedding_model,
    embedding_dim,
    chunker_version,
    model_hash,
    chunks,                # [{canonical_text, source_uri, byte_start, byte_end, embedding}]
    reproducible=True,
)
```

or via Pipeline in python/builder.py with chunker, SQLite cache and auto-validate.

v1 contract

flat exact search. real cosine score. recall 1.0. L2-normalized float32 embeddings. six required sections, 64-byte aligned, all checksummed. file_hash over the whole file, content_hash over the canonical sections, stable across rebuilds.

builds with reproducible=True are byte-identical for the same input.

tests

```
cargo test --workspace
python tests/test_e2e.py
python tests/test_builder.py
```

reference

ARCHITECTURE.md for binary layout, API, errors, versioning.
SPEC.md for the byte-by-byte format.

MIT.
