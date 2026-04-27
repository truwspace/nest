"""Convert a legacy SQLite-based truw_ptbr.nest into the new v1 binary format.

Legacy layout:
  - articles(id, block_id, pos_in_block, source, label)
  - text_blocks(block_id, data: zstd[ u32 n_texts | n_texts * u32 offset | body ])
  - blobs(name, data) with name in {manifest, faiss_index, embeddings}
    - embeddings: zstd[ float16 (N, D) ]

New v1 .nest:
  - manifest.embedding_model from legacy manifest
  - one chunk per article, canonical_text = article body
  - synthetic source_uri = `legacy://truw_ptbr/<id>`, byte span = [0, len(utf8(text))]
  - provenance carries the legacy labels/sources so nothing is lost
"""
from __future__ import annotations

import argparse
import json
import os
import struct
import sqlite3
import subprocess
import sys
import time

import numpy as np
import zstandard as zstd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import nest


def load_blocks(conn: sqlite3.Connection):
    """Yield (block_id, [text, ...]) for every text_block, in block_id order."""
    dec = zstd.ZstdDecompressor()
    for block_id, data in conn.execute(
        "SELECT block_id, data FROM text_blocks ORDER BY block_id"
    ):
        raw = dec.decompress(data)
        n = struct.unpack_from("<I", raw, 0)[0]
        offsets = struct.unpack_from(f"<{n}I", raw, 4)
        body = raw[4 + n * 4 :]
        texts: list[str] = []
        for i in range(n):
            start = offsets[i]
            end = offsets[i + 1] if i + 1 < n else len(body)
            texts.append(body[start:end].decode("utf-8"))
        yield block_id, texts


def load_embeddings(conn: sqlite3.Connection, n: int, dim: int) -> np.ndarray:
    dec = zstd.ZstdDecompressor()
    row = conn.execute("SELECT data FROM blobs WHERE name='embeddings'").fetchone()
    if row is None:
        raise SystemExit("legacy file has no `embeddings` blob")
    raw = dec.decompress(row[0])
    arr = np.frombuffer(raw, dtype=np.float16).reshape(n, dim).astype(np.float32)
    # The legacy builder stored unit-normalized vectors but float16 round-trip
    # can drift the norm by ~1e-3. Re-normalize so the runtime's dot product
    # is exactly cosine.
    norms = np.linalg.norm(arr, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return arr / norms


def convert(src: str, dst: str, *, reproducible: bool, validate_bin: str) -> None:
    if not os.path.exists(src):
        raise SystemExit(f"source not found: {src}")

    t0 = time.time()
    conn = sqlite3.connect(f"file:{src}?mode=ro", uri=True)
    conn.row_factory = sqlite3.Row

    legacy_manifest = json.loads(
        conn.execute("SELECT data FROM blobs WHERE name='manifest'").fetchone()["data"]
    )
    n = legacy_manifest["n_articles"]
    dim = legacy_manifest["embedding_dim"]
    model = legacy_manifest["embedding_model"]
    print(f"legacy: {n} articles, dim={dim}, model={model}")

    # Load all texts into memory, indexed by article id.
    texts: list[str | None] = [None] * n
    for block_id, block_texts in load_blocks(conn):
        for pos, text in enumerate(block_texts):
            aid = block_id * 256 + pos
            if aid < n:
                texts[aid] = text
    missing = [i for i, t in enumerate(texts) if t is None]
    if missing:
        raise SystemExit(f"{len(missing)} articles have no text (e.g. id={missing[:5]})")

    # Per-article metadata.
    rows = list(
        conn.execute(
            "SELECT id, source, label FROM articles ORDER BY id"
        )
    )
    if len(rows) != n:
        raise SystemExit(f"articles table has {len(rows)} rows but manifest says {n}")

    print("loading embeddings...")
    embs = load_embeddings(conn, n, dim)

    chunks = []
    labels = []
    sources = []
    for r in rows:
        aid = r["id"]
        text = texts[aid]
        ub = text.encode("utf-8")
        chunks.append(
            dict(
                canonical_text=text,
                source_uri=f"legacy://truw_ptbr/{aid}",
                byte_start=0,
                byte_end=len(ub),
                embedding=embs[aid].tolist(),
            )
        )
        labels.append(r["label"])
        sources.append(r["source"])

    provenance = {
        "legacy_source": "truw_ptbr.nest (SQLite)",
        "legacy_version": legacy_manifest.get("version", "unknown"),
        "legacy_created": legacy_manifest.get("created"),
        "labels": labels,
        "sources": sources,
        "note": "synthetic source_uri = legacy://truw_ptbr/<id> since the legacy format did not preserve original URIs",
    }

    # The legacy embedding model (paraphrase-multilingual-MiniLM-L12-v2) doesn't
    # ship with a stable reproducible hash here, so we tag it as a placeholder
    # zero-hash. Callers that need a verified model_hash can override.
    model_hash = "sha256:" + "0" * 64

    if os.path.exists(dst):
        os.unlink(dst)
    nest.build(
        output_path=dst,
        embedding_model=model,
        embedding_dim=dim,
        chunker_version="legacy/truw_ptbr_v0.1.0",
        model_hash=model_hash,
        chunks=chunks,
        title="truw_ptbr",
        version="v1-from-legacy",
        description="Portuguese fake-news corpus, converted from the legacy SQLite-based .nest",
        license=legacy_manifest.get("license"),
        provenance=provenance,
        reproducible=reproducible,
    )
    elapsed = time.time() - t0
    size = os.path.getsize(dst)
    print(f"wrote {dst}: {size/1e6:.2f} MB in {elapsed:.1f}s")

    # Final integrity check via the CLI.
    out = subprocess.run([validate_bin, "validate", dst], capture_output=True, text=True)
    if out.returncode != 0:
        raise SystemExit(f"nest validate failed:\n{out.stderr}\n{out.stdout}")
    print(out.stdout.strip())


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--src", default="data/truw_ptbr.nest")
    p.add_argument("--dst", default="data/truw_ptbr.v1.nest")
    p.add_argument("--reproducible", action="store_true")
    p.add_argument(
        "--nest-bin",
        default=os.path.join(os.path.dirname(__file__), "..", "target", "release", "nest"),
    )
    args = p.parse_args()
    convert(args.src, args.dst, reproducible=args.reproducible, validate_bin=args.nest_bin)


if __name__ == "__main__":
    main()
