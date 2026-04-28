"""Latency-bench helpers used by `measure_presets.py`.

Pure functions over a `nest.NestFile` plus a list of `(qvec, qtext)`
queries — no `.nest` I/O, no result formatting. Internal to
`python/tools/`.
"""

from __future__ import annotations

import time
from pathlib import Path


def percentile(values: list[float], p: float) -> float:
    if not values:
        return float("nan")
    s = sorted(values)
    idx = min(len(s) - 1, max(0, round((len(s) - 1) * p)))
    return s[idx]


def run_bench(
    db,
    queries,
    k: int,
    mode: str,
    ef: int = 100,
    candidates: int = 200,
):
    """Time `len(queries)` invocations of the requested search mode.

    Returns `(times_ms, hits_per_query)`. Caller computes recall against
    a baseline `hits_per_query` and percentiles over `times_ms`.
    """
    times: list[float] = []
    results = []
    for qvec, qtext in queries:
        t0 = time.time()
        if mode == "exact":
            hits = db.search(qvec, k)
        elif mode == "ann":
            hits = db.search_ann(qvec, k, ef)
        elif mode == "hybrid":
            hits = db.search_hybrid(qvec, qtext, k, candidates)
        else:
            raise ValueError(mode)
        dt = (time.time() - t0) * 1000.0
        times.append(dt)
        results.append(hits)
    return times, results


def build_variant(chunks, meta, preset: str, out_path: Path):
    """Build `out_path` with the given preset; return seconds elapsed.

    Imports `nest` lazily because `_bench_runner` is meant to be cheap
    to import (unlike the PyO3 extension load, which pulls a 1.6 MB .so).
    """
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    import nest

    if out_path.exists():
        out_path.unlink()
    t0 = time.time()
    nest.build(
        output_path=str(out_path),
        embedding_model=meta["embedding_model"],
        embedding_dim=meta["embedding_dim"],
        chunker_version=meta["chunker_version"],
        model_hash=meta["model_hash"],
        chunks=chunks,
        reproducible=True,
        preset=preset,
    )
    return time.time() - t0
