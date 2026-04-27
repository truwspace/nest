Não escreva markdowns, memórias ou arquivos de documentação NÃO SOLICITADOS.

# Build & Test

- `cargo build --workspace` / `cargo build --release --workspace`
- `cargo test --workspace` — runs all Rust tests (unit + integration + golden)
- `cargo fmt --all --check` — formatting check
- `cargo clippy --workspace -- -D warnings` — linting (warnings are errors)
- `ruff check .` / `ruff format .` — Python linting/formatting (config in `pyproject.toml`)

# PyO3 Extension

No maturin. Build manually:

```
cargo build --release -p nest-python
cp target/release/lib_nest.dylib python/_nest.so   # macOS
# cp target/release/lib_nest.so python/_nest.so     # Linux
```

abi3 targets Python 3.12+ (NOT 3.14). Python tests need the built `.so` first:

```
python tests/test_e2e.py
python tests/test_builder.py
```

No pytest — tests are plain scripts with `if __name__ == "__main__"`.

# Single-Target Commands

- `cargo test -p nest-format` — format crate only
- `cargo test -p nest-runtime` — runtime crate only
- `cargo run -p nest-format --example regen_golden` — regenerate golden fixture
- `cargo test -p nest-cli` — CLI integration tests (requires release build)

# Architecture

```
nest-format  ← standalone library (binary format spec, reader, writer, manifest, hashing)
nest-runtime ← depends on nest-format (mmap-backed search, MmapNestFile)
nest-cli     ← depends on nest-format + nest-runtime (clap binary, subcommands)
nest-python  ← depends on nest-format + nest-runtime (cdylib _nest, PyO3 abi3-py312)
```

CLI binary: `nest` (subcommands: inspect, validate, search, benchmark, stats, cite).
Python entry: `sys.path.insert(0, "python"); import nest` — dynamic loader finds `_nest.so`/`lib_nest.dylib`.

# Conventions

- Rust edition 2024, resolver 3, `thiserror` for errors (never panic in library code).
- `repr(C)` structs for binary layout; all integers LE unsigned.
- Binary format v1 is frozen — bump `NEST_FORMAT_VERSION` for breaking changes.
- Hash format: always `sha256:<64 lowercase hex>`.
- `NestFileBuilder` is a consuming builder (`add_chunk(self) -> Self`).
- Golden fixture: `crates/nest-format/tests/fixtures/golden_v1_minimal.nest` (1366 bytes).
- CLI `search` takes JSON f32 array positional arg; no bundled embedding model.
- `CONTRIBUTING.md` is an unfilled template — ignore it.