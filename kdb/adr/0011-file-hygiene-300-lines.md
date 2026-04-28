# ADR 0011: File hygiene — 300 lines per source file

- **Status:** Accepted
- **Date:** 2026-04-28
- **Deciders:** project owner

## Context

Phase 5 of the 2026-04-28 push refactored 12 Rust files that had
grown past 300 lines each (largest: `ann.rs` at 698 lines, `writer.rs`
at 626, `runtime/lib.rs` at 572). The refactor was triggered by a
direct policy request: "max 300 linhas por arquivo".

Three reasons for the rule, in priority order:

1. **Bisection.** A 698-line module on `git bisect` reduces the
   confidence of a regression introduction to a 698-line range.
   Splitting into 5 modules of ≤ 200 lines each gives bisect a
   meaningful per-commit blast radius.
2. **Single Responsibility Principle.** Files that grew past 300
   lines invariably mixed two concerns: e.g. `ann.rs` had build,
   search, codec, and neighbor selection in one file. Splitting
   forced the split-by-concern that should have happened originally.
3. **Code review surface.** A reviewer reads ~300 lines per pass
   without losing the call graph. Past that, comments degrade.

Costs we accepted:

- More files. Phase 5 turned 12 monoliths into ~50 modules. Tab
  switching is the new bottleneck, not scrolling.
- More `mod.rs` boilerplate. Each split needs a new `mod.rs` with
  re-exports.

## Decision

Every Rust file under `crates/**/src/**` (excluding `tests/`) must be
≤ 300 lines. `release_check.sh` enforces this with:

```bash
find crates -name '*.rs' -not -path '*/tests/*' \
  | xargs wc -l \
  | awk '$1 > 300 {print}' \
  | grep -v 'total$'
```

If the output is non-empty, the release check fails.

The same rule applies to first-party Python modules:
`python/*.py`, `python/tools/*.py`, `tests/*.py`. Tools may grow up
to 300 lines before splitting; helpers extract to private modules
prefixed `_` (e.g. `_baseline_decoder.py`).

Excluded:

- Test files (`crates/**/tests/*.rs`, `tests/*.py`). Tests can grow
  up to 600 lines because they're flat lists of `#[test]` cases. The
  bisection argument doesn't apply — a test file rarely contains a
  bug, and growth is by-design.
- Generated files (`Cargo.lock`, build artifacts).

## Consequences

### Positive

- The codebase has 0 files > 300 lines under `crates/**/src` as of
  v0.2.0. New code must hold the line.
- The split forced by Phase 5 produced clearly-named modules that
  match the domain vocabulary: `select_neighbors.rs`,
  `materialize.rs`, `embeddings_payload.rs` (in writer/), etc.

### Negative

- More mod.rs files. ~50 new modules in Phase 5 added ~250 lines of
  re-export boilerplate. The cost is real but small.
- Some splits feel artificial when a single concern is just slightly
  too long. We pick the cleanest seam available even when it leaves
  one file at, say, 270 lines.

### Trade-offs

- Did **not** apply the rule to `crates/**/tests/*.rs`. Test files
  routinely hit 600+ lines as `#[test]` lists; splitting them by
  topic adds friction without buying bisectability.
- Did **not** lower the threshold to 200 lines. 300 is a good
  empirical "fits a screen plus context" line; lower forces
  artificial splits.

## Alternatives considered

- **400-line threshold.** Rejected: empirically catches one or two
  fewer monoliths but lets the SRP-violator class through (most
  500-line files are doing two things).
- **Lines-of-code (LOC) instead of physical lines.** Rejected:
  comments + doc are useful, removing them from the count incentivizes
  removing them. Physical lines is the dumb-but-honest metric.
- **Bytes (200KB) threshold.** Rejected: same metric in disguise,
  worse to scan visually.

## References

- `scripts/release_check.sh` (enforces the gate).
- Phase 5 commits: `787dff3b`, `95d7da7c`, `233e81db`.
- `crates/nest-runtime/src/ann/` (5-module split, the largest).
- `crates/nest-runtime/src/{search,mmap_file,materialize}.rs` (
  split of the 572-line `lib.rs`).
