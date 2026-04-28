# ADR 0004: PyO3 is the only Python entry point

- **Status:** Accepted
- **Date:** 2026-04-27
- **Deciders:** project owner

## Context

Two Python paths into the runtime were possible:

1. **PyO3** (`python/_nest.so`): in-process call into Rust.
2. **Subprocess** (`subprocess.run([NEST_BIN, "validate", path])`):
   shell out to the CLI binary.

Early prototypes had Python helpers using both — PyO3 for fast paths,
subprocess for `nest validate` after a build. That is two paths to
maintain, two failure modes to debug, and a hidden dependency on
`target/release/nest` being on disk.

## Decision

`python/` has exactly one Python entry point: PyO3. The `nest.NestFile`
class exposes everything a Python caller needs — `open`, `search`,
`inspect`, `validate`, plus the `nest.build` / `nest.chunk_id`
free functions. There is no subprocess fallback inside `python/` or
`python/builder.py` or `python/convert_legacy.py`.

The CLI binary continues to exist and is exercised by
`crates/nest-cli/tests/cli_e2e.rs`. Python tests do not shell out to it.

## Consequences

### Positive

- A single failure mode for Python users: an exception from PyO3.
- `python/builder.py::Pipeline.emit()` is portable to any host that
  has the cdylib loaded; the Rust binary need not be on disk.
- No PATH ambiguity ("which `nest` am I running?").

### Negative

- The `nest cite` command has no PyO3 equivalent (cite output is
  human-formatted text; the underlying `NestFile.inspect()` exposes the
  raw fields). Python callers wanting the formatted cite output must
  shell out themselves — explicitly, outside the library.

### Trade-offs

- Slightly more PyO3 surface to maintain (the `inspect_json` and
  `revalidate` Rust helpers that back `NestFile.inspect` /
  `NestFile.validate`). Worth it.

## Alternatives considered

- **Keep both paths; pick whichever is faster per call site.**
  Rejected — split surfaces invite divergence (the subprocess `validate`
  silently re-validates with a different binary version than the PyO3
  reader).

## References

- `crates/nest-python/src/lib.rs`
- `python/nest.py`
- `python/builder.py` (Pipeline.emit uses `nest.open(p).validate()`)
- `python/convert_legacy.py` (uses PyO3 path for validation)
