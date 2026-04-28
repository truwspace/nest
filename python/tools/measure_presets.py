"""Acceptance harness: measure file size, recall@k, score drift, and
latency for the four presets against a baseline `data/corpus_next.v1.nest`.

Pipeline:

  1. Open the baseline (`exact` preset) — this is the recall=1.0 ground
     truth for both ranking and score.
  2. Decode embeddings + canonical texts + chunk_ids out of the baseline.
  3. For each variant in {compressed, tiny, hybrid}, rebuild a .nest with
     the same corpus and the variant preset.
  4. For N random queries (sampled from the corpus' own embeddings —
     deterministic given a seed):
       - exact (baseline) top-k
       - variant top-k (exact / ann / hybrid as appropriate)
       - recall@k = |overlap| / k
       - score drift = |variant_score[0] - exact_score[0]| (top-hit)
  5. Latency: per-variant p50/p95/p99 over the same queries.

Output: a markdown table to stdout. No artifacts persisted beyond the
variant `.nest` files (under `data/measure/`) so the run is rerunnable.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from pathlib import Path
from statistics import mean

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "python"))
import nest  # noqa: E402

DEFAULT_BASELINE = REPO / "data" / "corpus_next.v1.nest"
OUT_DIR = REPO / "data" / "measure"


def _decode_baseline(path: Path):
    """Pull (chunks, manifest_meta) from a baseline so we can rebuild it
    at a different preset. Returns the chunks list ready for nest.build()."""
    db = nest.open(str(path))
    info = db.inspect()
    n = db.n_embeddings
    dim = db.embedding_dim
    print(f"baseline: {path}  n={n} dim={dim} dtype={db.dtype}", file=sys.stderr)

    # Re-derive embeddings: easier to query the unit-vector basis we want
    # via search? No — we need the actual stored vectors. Easiest path:
    # use the in-process exact search to score every chunk against itself
    # and recover the vectors via the section. But the runtime hides
    # those. So we shell out to the binary reader path: read the file,
    # parse the embeddings section as float32 LE.
    raw = path.read_bytes()
    section_table_offset = int.from_bytes(raw[40:48], "little")
    section_table_count = int.from_bytes(raw[48:56], "little")
    embeddings_payload = None
    canonical_payload = None
    spans_payload = None
    chunk_ids_payload = None
    for i in range(section_table_count):
        eoff = section_table_offset + i * 32
        sid = int.from_bytes(raw[eoff : eoff + 4], "little")
        enc = int.from_bytes(raw[eoff + 4 : eoff + 8], "little")
        off = int.from_bytes(raw[eoff + 8 : eoff + 16], "little")
        size = int.from_bytes(raw[eoff + 16 : eoff + 24], "little")
        payload = raw[off : off + size]
        if enc != 0:
            raise SystemExit(f"baseline must be raw-encoded; section 0x{sid:02x} encoding={enc}")
        if sid == 0x04:
            embeddings_payload = payload
        elif sid == 0x02:
            canonical_payload = payload
        elif sid == 0x03:
            spans_payload = payload
        elif sid == 0x01:
            chunk_ids_payload = payload
    assert embeddings_payload and canonical_payload and spans_payload and chunk_ids_payload

    # Decode embeddings: f32 LE, n*dim
    import struct

    embs = list(struct.iter_unpack("<f", embeddings_payload))
    embs = [e[0] for e in embs]

    # Decode canonical texts: u32 version | u64 count | (u32 len, bytes)*
    def _decode_strings(buf, expected):
        pos = 0
        ver = struct.unpack_from("<I", buf, pos)[0]
        pos += 4
        cnt = struct.unpack_from("<Q", buf, pos)[0]
        pos += 8
        assert ver == 1 and cnt == expected, (ver, cnt, expected)
        out = []
        for _ in range(cnt):
            slen = struct.unpack_from("<I", buf, pos)[0]
            pos += 4
            out.append(buf[pos : pos + slen].decode("utf-8"))
            pos += slen
        return out

    texts = _decode_strings(canonical_payload, n)

    # Decode spans: prefix + (u32 len, bytes uri, u64 start, u64 end)*
    pos = 0
    ver = struct.unpack_from("<I", spans_payload, pos)[0]
    pos += 4
    cnt = struct.unpack_from("<Q", spans_payload, pos)[0]
    pos += 8
    assert ver == 1 and cnt == n
    spans = []
    for _ in range(n):
        slen = struct.unpack_from("<I", spans_payload, pos)[0]
        pos += 4
        uri = spans_payload[pos : pos + slen].decode("utf-8")
        pos += slen
        start = struct.unpack_from("<Q", spans_payload, pos)[0]
        pos += 8
        end = struct.unpack_from("<Q", spans_payload, pos)[0]
        pos += 8
        spans.append((uri, start, end))

    chunks = []
    for i in range(n):
        chunks.append(
            dict(
                canonical_text=texts[i],
                source_uri=spans[i][0],
                byte_start=spans[i][1],
                byte_end=spans[i][2],
                embedding=embs[i * dim : (i + 1) * dim],
            )
        )
    meta = dict(
        embedding_model=info["manifest"]["embedding_model"],
        embedding_dim=dim,
        chunker_version=info["manifest"]["chunker_version"],
        model_hash=info["manifest"]["model_hash"],
    )
    return chunks, meta


def _build_variant(chunks, meta, preset: str, out_path: Path) -> float:
    """Build out_path with the given preset, return seconds elapsed."""
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


def _percentile(values, p):
    if not values:
        return float("nan")
    s = sorted(values)
    idx = min(len(s) - 1, max(0, round((len(s) - 1) * p)))
    return s[idx]


def _bench(db, queries, k, mode: str, ef: int = 100, candidates: int = 200):
    times = []
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--baseline", default=str(DEFAULT_BASELINE))
    ap.add_argument("--n-queries", type=int, default=50)
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--variants", default="compressed,tiny,hybrid")
    ap.add_argument(
        "--json",
        action="store_true",
        help="Emit results as JSON to stdout (table to stderr) for regression gates.",
    )
    args = ap.parse_args()

    base_path = Path(args.baseline)
    if not base_path.exists():
        raise SystemExit(f"baseline not found: {base_path}")

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    chunks, meta = _decode_baseline(base_path)
    n = len(chunks)
    dim = meta["embedding_dim"]

    # Sample queries: pick `n_queries` chunks at random, use their
    # embedding as the query (slightly tweaked) and their canonical text
    # for BM25.
    rng = random.Random(args.seed)
    pool = rng.sample(range(n), args.n_queries)
    queries = []
    for i in pool:
        # Use the chunk's own embedding so search returns the chunk
        # itself with score≈1; small perturbation so it's not trivial.
        qvec = list(chunks[i]["embedding"])
        # add a tiny noise (deterministic)
        for j in range(dim):
            qvec[j] += ((j * 7 + i) % 17 - 8) * 1e-5
        # re-normalize
        norm = sum(x * x for x in qvec) ** 0.5 or 1.0
        qvec = [x / norm for x in qvec]
        qtext = chunks[i]["canonical_text"][:200]
        queries.append((qvec, qtext))

    # 1. Baseline (exact).
    db_exact = nest.open(str(base_path))
    base_size = base_path.stat().st_size
    t_exact, hits_exact = _bench(db_exact, queries, args.k, mode="exact")
    base_top_score = [h[0].score for h in hits_exact]

    log = sys.stderr if args.json else sys.stdout
    print(file=log)
    print(
        f"## Acceptance harness: {n} chunks, dim={dim}, {args.n_queries} queries, k={args.k}",
        file=log,
    )
    print(f"baseline:           {base_path}", file=log)
    print(f"baseline file_hash: {db_exact.file_hash}", file=log)
    print(f"baseline simd:      {db_exact.simd_backend}", file=log)
    print(file=log)
    rows = []
    rows.append(
        (
            "preset",
            "size_MB",
            "size_ratio",
            "build_s",
            "recall@k",
            "drift_max",
            "drift_mean",
            "p50_ms",
            "p95_ms",
            "p99_ms",
        )
    )
    rows.append(
        (
            "exact (baseline)",
            f"{base_size / 1e6:.2f}",
            "1.000",
            "—",
            "1.0000",
            "0.000000",
            "0.000000",
            f"{_percentile(t_exact, 0.50):.3f}",
            f"{_percentile(t_exact, 0.95):.3f}",
            f"{_percentile(t_exact, 0.99):.3f}",
        )
    )
    measurements = [
        {
            "preset": "exact",
            "is_baseline": True,
            "size_mb": round(base_size / 1e6, 4),
            "size_ratio": 1.0,
            "build_s": None,
            "recall_at_k": 1.0,
            "drift_max": 0.0,
            "drift_mean": 0.0,
            "p50_ms": round(_percentile(t_exact, 0.50), 4),
            "p95_ms": round(_percentile(t_exact, 0.95), 4),
            "p99_ms": round(_percentile(t_exact, 0.99), 4),
            "dtype": db_exact.dtype,
            "has_ann": db_exact.has_ann,
            "has_bm25": db_exact.has_bm25,
            "search_mode": "exact",
            "file_hash": db_exact.file_hash,
            "content_hash": db_exact.content_hash,
        }
    ]

    for preset in args.variants.split(","):
        preset = preset.strip()
        if not preset:
            continue
        out_path = OUT_DIR / f"corpus_{preset}.nest"
        print(f"\n→ building preset={preset} → {out_path}", file=log)
        build_time = _build_variant(chunks, meta, preset, out_path)
        size = out_path.stat().st_size
        size_ratio = size / base_size

        db_v = nest.open(str(out_path))
        # Validate.
        db_v.validate()

        # Search mode: ann if hnsw present, hybrid if both.
        if db_v.has_ann and db_v.has_bm25 and preset == "hybrid":
            mode = "hybrid"
        elif db_v.has_ann and preset == "tiny":
            mode = "ann"
        else:
            mode = "exact"
        print(
            f"  size={size / 1e6:.2f} MB ({size_ratio:.3f}× baseline)  "
            f"dtype={db_v.dtype}  has_ann={db_v.has_ann}  has_bm25={db_v.has_bm25}  "
            f"search_mode={mode}  build={build_time:.1f}s",
            file=log,
        )

        t_v, hits_v = _bench(db_v, queries, args.k, mode=mode)

        # Recall@k against exact baseline.
        recalls = []
        drifts = []
        for h_exact, h_v, base_score in zip(hits_exact, hits_v, base_top_score, strict=False):
            ex_ids = {h.chunk_id for h in h_exact}
            v_ids = {h.chunk_id for h in h_v}
            recalls.append(len(ex_ids & v_ids) / args.k)
            if h_v:
                drifts.append(abs(h_v[0].score - base_score))
        recall = mean(recalls) if recalls else 0.0
        drift_max = max(drifts) if drifts else 0.0
        drift_mean = mean(drifts) if drifts else 0.0

        rows.append(
            (
                preset,
                f"{size / 1e6:.2f}",
                f"{size_ratio:.3f}",
                f"{build_time:.1f}",
                f"{recall:.4f}",
                f"{drift_max:.6f}",
                f"{drift_mean:.6f}",
                f"{_percentile(t_v, 0.50):.3f}",
                f"{_percentile(t_v, 0.95):.3f}",
                f"{_percentile(t_v, 0.99):.3f}",
            )
        )
        measurements.append(
            {
                "preset": preset,
                "is_baseline": False,
                "size_mb": round(size / 1e6, 4),
                "size_ratio": round(size_ratio, 4),
                "build_s": round(build_time, 2),
                "recall_at_k": round(recall, 4),
                "drift_max": round(drift_max, 6),
                "drift_mean": round(drift_mean, 6),
                "p50_ms": round(_percentile(t_v, 0.50), 4),
                "p95_ms": round(_percentile(t_v, 0.95), 4),
                "p99_ms": round(_percentile(t_v, 0.99), 4),
                "dtype": db_v.dtype,
                "has_ann": db_v.has_ann,
                "has_bm25": db_v.has_bm25,
                "search_mode": mode,
                "file_hash": db_v.file_hash,
                "content_hash": db_v.content_hash,
            }
        )

    print(file=log)
    print("## Results", file=log)
    widths = [max(len(str(r[i])) for r in rows) for i in range(len(rows[0]))]
    sep = "  ".join("-" * w for w in widths)
    for i, row in enumerate(rows):
        line = "  ".join(str(c).ljust(w) for c, w in zip(row, widths, strict=False))
        print(line, file=log)
        if i == 0:
            print(sep, file=log)

    if args.json:
        doc = {
            "schema_version": 1,
            "n_chunks": n,
            "embedding_dim": dim,
            "n_queries": args.n_queries,
            "k": args.k,
            "seed": args.seed,
            "baseline_path": str(base_path),
            "baseline_file_hash": db_exact.file_hash,
            "baseline_content_hash": db_exact.content_hash,
            "simd_backend": db_exact.simd_backend,
            "presets": measurements,
        }
        json.dump(doc, sys.stdout, indent=2, sort_keys=True)
        sys.stdout.write("\n")


if __name__ == "__main__":
    main()
