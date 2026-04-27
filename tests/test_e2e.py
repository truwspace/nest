"""End-to-end test: build a .nest from Python, then search via Python and CLI."""
import os
import sys
import json
import math
import random
import tempfile
import subprocess

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "python"))

import nest

NEST_BIN = os.path.join(
    os.path.dirname(__file__), "..", "target", "release", "nest"
)


def _unit_vec(rng: random.Random, dim: int) -> list[float]:
    v = [rng.random() for _ in range(dim)]
    n = math.sqrt(sum(x * x for x in v)) or 1.0
    return [x / n for x in v]


def make_nest(path: str, dim: int, n: int, *, reproducible: bool = False, seed: int = 0):
    rng = random.Random(seed)
    chunks = []
    cursor = 0
    for i in range(n):
        text = f"chunk_{i}"
        chunks.append(
            dict(
                canonical_text=text,
                source_uri="doc.txt",
                byte_start=cursor,
                byte_end=cursor + len(text),
                embedding=_unit_vec(rng, dim),
            )
        )
        cursor += len(text)
    nest.build(
        output_path=path,
        embedding_model="test-model",
        embedding_dim=dim,
        chunker_version="test-chunker/1",
        model_hash="sha256:" + "0" * 64,
        chunks=chunks,
        reproducible=reproducible,
    )


def test_python_build_then_python_search():
    with tempfile.NamedTemporaryFile(delete=False, suffix=".nest") as f:
        path = f.name
    try:
        make_nest(path, dim=8, n=10)
        db = nest.open(path)
        assert db.embedding_dim == 8
        assert db.n_embeddings == 10
        assert db.file_hash.startswith("sha256:")
        assert db.content_hash.startswith("sha256:")

        q = [1.0] + [0.0] * 7
        hits = db.search(q, 3)
        assert len(hits) == 3
        h = hits[0]
        assert h.score_type == "cosine"
        assert h.index_type == "exact"
        assert h.reranked is False
        assert h.file_hash == db.file_hash
        assert h.content_hash == db.content_hash
        assert h.citation_id.startswith(f"nest://{db.content_hash}/")
        print("python build/search OK:", path)
    finally:
        os.unlink(path)


def test_python_build_then_cli():
    with tempfile.NamedTemporaryFile(delete=False, suffix=".nest") as f:
        path = f.name
    try:
        make_nest(path, dim=4, n=5)

        out = subprocess.run([NEST_BIN, "validate", path], capture_output=True, text=True)
        assert out.returncode == 0, out.stderr
        assert "valid .nest v1 file" in out.stdout

        out = subprocess.run([NEST_BIN, "stats", path], capture_output=True, text=True)
        assert out.returncode == 0, out.stderr
        assert "chunks:       5" in out.stdout
        assert "metric:       ip" in out.stdout

        q = json.dumps([1.0, 0.0, 0.0, 0.0])
        out = subprocess.run(
            [NEST_BIN, "search", path, q, "-k", "1"], capture_output=True, text=True
        )
        assert out.returncode == 0, out.stderr
        assert "citation_id=nest://sha256:" in out.stdout
        print("cli validate/stats/search OK")
    finally:
        os.unlink(path)


def test_reproducible_builds_match_byte_for_byte():
    with tempfile.TemporaryDirectory() as d:
        a = os.path.join(d, "a.nest")
        b = os.path.join(d, "b.nest")
        make_nest(a, dim=4, n=3, reproducible=True, seed=7)
        make_nest(b, dim=4, n=3, reproducible=True, seed=7)
        with open(a, "rb") as fa, open(b, "rb") as fb:
            data_a = fa.read()
            data_b = fb.read()
        assert data_a == data_b, "reproducible builds diverged"
        print("reproducible build OK:", len(data_a), "bytes")


def test_cite_resolves_citation():
    with tempfile.NamedTemporaryFile(delete=False, suffix=".nest") as f:
        path = f.name
    try:
        make_nest(path, dim=4, n=3)
        db = nest.open(path)
        hits = db.search([1.0, 0.0, 0.0, 0.0], 1)
        cit = hits[0].citation_id

        out = subprocess.run(
            [NEST_BIN, "cite", path, cit], capture_output=True, text=True
        )
        assert out.returncode == 0, out.stderr
        assert "source_uri:" in out.stdout
        assert "byte_start:" in out.stdout
        assert "text:" in out.stdout
        print("cite OK")
    finally:
        os.unlink(path)


if __name__ == "__main__":
    test_python_build_then_python_search()
    test_python_build_then_cli()
    test_reproducible_builds_match_byte_for_byte()
    test_cite_resolves_citation()
