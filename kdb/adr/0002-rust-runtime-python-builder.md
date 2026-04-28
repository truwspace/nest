# ADR 0002: Rust runtime, Python builder

- **Status:** Accepted
- **Date:** 2026-04-27
- **Deciders:** project owner

## Context

Two halves of the system have different requirements:

- **Read path** (open, mmap, search, validate, cite) must be
  predictable, low-latency, dependency-free, and shippable as a single
  binary alongside any agent runtime.
- **Build path** (text ingestion, scraping, chunking, embedding model
  inference) is dominated by Python ecosystem libraries
  (`sentence-transformers`, `huggingface_hub`, `numpy`, `zstandard`)
  that have no Rust equivalent.

A pure-Rust build path would require re-implementing or FFI-binding a
half-dozen ML stacks. A pure-Python read path would forfeit the
mmap-friendly latency story and add Python as a runtime dependency for
agents that just want to consume `.nest` files.

## Decision

Split along the read/write boundary:

- **Rust** owns: `nest-format` (codecs, layout, hashes), `nest-runtime`
  (mmap-backed search), `nest-cli` (`inspect/validate/stats/search/
  benchmark/cite`).
- **Python** owns: ingestion, chunking, embedding, builder pipeline.
  The Python side calls into Rust via PyO3 (`crates/nest-python`) for
  `nest.build / open / search / inspect / validate / chunk_id`. No
  Python re-implementation of the format.

A `.nest` file is the only contract between halves. Either side can be
swapped (e.g. a TypeScript builder, a Go reader) without breaking the
other.

## Consequences

### Positive

- Agent runtimes get a native, dependency-free reader binary.
- Builder authors get the entire Python ML ecosystem.
- The format is the single source of truth — neither half is a
  dependency of the other at runtime.

### Negative

- Two languages, two toolchains, two test harnesses.
- Python developers must compile a cdylib (`cargo build --release -p
  nest-python && cp target/release/lib_nest.dylib python/_nest.so`)
  before importing `nest`. There is no `pip install`.

### Trade-offs

- **No `maturin develop` / wheel publication** in v0.1.0 — the cdylib
  copy step is one command and avoids committing to a packaging
  cadence we do not yet need.
- **No `nest build` CLI subcommand** — building requires a model and a
  chunker, both of which Python ships better.

## Alternatives considered

- **Pure Rust, including embedding inference (e.g. via candle or
  ort).** Rejected: model coverage is far behind Python, and embedding
  is not where v0.1.0 needs to differentiate.
- **Pure Python, with a Rust-style binary format implemented in
  ctypes/numpy.** Rejected: no zero-copy mmap story; agent runtimes
  would need Python at runtime.

## References

- `crates/nest-runtime/src/lib.rs`
- `crates/nest-python/src/lib.rs`
- `python/builder.py`
- `python/convert_legacy.py`
