# ADR 0009: SIMD dispatch at runtime, not compile time

- **Status:** Accepted
- **Date:** 2026-04-28
- **Deciders:** project owner

## Context

The dot-product hot path runs once per chunk per query — for the
30k-chunk corpus that's 30,000 multiply-adds per query in exact
flat search. Auto-vectorization gets us part of the way (3 ms p50
on Apple Silicon), but explicit AVX2 / NEON intrinsics drop p50 to
1 ms on int8 and ~3 ms on f32, a 3× speedup hot-cache.

We have to support three dtypes (f32, f16, i8) and at least three
backends (AVX2 on x86_64, NEON on aarch64, scalar fallback for
everything else). Compile-time `#[cfg]` selection forces a per-target
build matrix and bricks the binary on hosts whose CPU advertises but
doesn't reliably implement the feature.

## Decision

Dispatch is runtime, with a one-time CPUID-style probe cached in a
`OnceLock<SimdBackend>`:

- `x86_64`: AVX2 if `is_x86_feature_detected!("avx2")` AND `"fma"`,
  else fall back to scalar.
- `aarch64`: NEON if `is_aarch64_feature_detected!("neon")`, else
  scalar (NEON is mandatory on Apple Silicon and any post-2018 ARM
  core, so this path is essentially "always taken").
- Anything else: scalar.

The dispatcher exposes `dot_f32_bytes`, `dot_f32_f16_bytes`, and
`dot_f32_i8`. Each calls into one of:
- `simd/scalar.rs` (auto-vectorizable fallback)
- `simd/avx2.rs` (`#[target_feature(enable = "avx2,fma")]`)
- `simd/neon.rs` (`#[target_feature(enable = "neon")]`)

All accumulators are f32 regardless of dtype. f16 is widened lane-by-
lane (`vcvt_f32_f16` on NEON). i8 is widened i8→i16→i32→f32 in
groups of 8 lanes.

`NEST_FORCE_SCALAR=1` short-circuits the probe to `Scalar`. Used for
before/after benchmarks on the same binary without recompilation.

Parity is asserted in `simd/tests.rs`: the dispatched backend must
agree with scalar to within 1e-4 (f32 / i8) or 1e-3 (f16, which loses
~5e-4 per component to rounding) over dim ∈ {4, 7, 16, 31, 64, 384}.

## Consequences

### Positive

- One binary works on every host. The CI matrix doesn't expand by
  arch.
- Users can A/B SIMD vs scalar without rebuilding (
  `NEST_FORCE_SCALAR=1 nest benchmark ...`). This was load-bearing
  during Phase 2's HNSW debugging.
- The format crate stays SIMD-agnostic. Embeddings sit in mmap as
  little-endian f32/f16/i8 bytes; the runtime decides how to
  multiply them.

### Negative

- One indirect call per dot product (the dispatcher). Negligible —
  the call is amortized over `dim` multiply-adds inside.
- `unsafe` is unavoidable in the AVX2/NEON paths (intrinsics are
  unsafe by default). Mitigation: each unsafe block has a
  `// SAFETY:` comment, parity tests run on every commit.

### Trade-offs

- Did **not** ship F16C-specific AVX2 paths. `_mm256_cvtph_ps`
  exists but the F16C feature flag is separate from AVX2's; runtime
  detection would add another branch. The scalar f16 path on x86 is
  fast enough.

## Alternatives considered

- **Compile-time `#[cfg]` only.** Rejected: forces per-arch builds;
  bricks unusual hosts.
- **One binary per arch via `cargo build --target`.** Rejected:
  multiplies CI cost; doesn't solve "this Xeon doesn't have AVX2".
- **Use a portable SIMD crate (`packed_simd`, `wide`).** Rejected:
  still need explicit f16/i8 widening; the wins are smaller than
  hand-tuning.

## References

- `crates/nest-runtime/src/simd/{mod,scalar,avx2,neon,tests}.rs`.
- `crates/nest-runtime/src/simd/mod.rs::detect_backend`.
- `nest benchmark --madvise-cold` for the latency contract.
