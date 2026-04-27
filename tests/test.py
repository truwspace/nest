"""Tests for nest — Nested Embedding Search Tool."""

import tempfile
from pathlib import Path

import numpy as np
import pytest

import nest


@pytest.fixture
def sample_db(tmp_path):
    """Create a small .nest for testing."""
    N, D = 100, 64
    rng = np.random.RandomState(42)
    embeddings = rng.randn(N, D).astype(np.float32)
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    embeddings = embeddings / norms

    texts = [f"Article {i} about topic {i % 10}" for i in range(N)]
    sources = [f"source_{i % 3}" for i in range(N)]
    labels = ["fake" if i % 2 == 0 else "true" for i in range(N)]

    path = str(tmp_path / "test.nest")
    nest.build(
        texts=texts,
        embeddings=embeddings,
        output_path=path,
        sources=sources,
        labels=labels,
        embedding_model="test-model",
        faiss_nlist=4,
        faiss_m=8,
        faiss_nbits=8,
    )
    return path, embeddings, texts


def test_build_creates_file(sample_db):
    path, _, _ = sample_db
    assert Path(path).exists()
    assert Path(path).stat().st_size > 0


def test_open_and_repr(sample_db):
    path, _, _ = sample_db
    db = nest.open(path)
    assert db.n_articles == 100
    assert "100 articles" in repr(db)
    db.close()


def test_manifest(sample_db):
    path, _, _ = sample_db
    db = nest.open(path)
    assert db.manifest["n_articles"] == 100
    assert db.manifest["embedding_dim"] == 64
    assert db.manifest["embedding_model"] == "test-model"
    db.close()


def test_search_by_embedding(sample_db):
    path, embeddings, texts = sample_db
    db = nest.open(path)
    results = db.search(embeddings[0], k=5)
    assert len(results) == 5
    assert results[0].id == 0
    assert results[0].text == texts[0]
    db.close()


def test_search_returns_searchresult(sample_db):
    path, embeddings, _ = sample_db
    db = nest.open(path)
    results = db.search(embeddings[0], k=1)
    r = results[0]
    assert isinstance(r, nest.SearchResult)
    assert isinstance(r.id, int)
    assert isinstance(r.score, float)
    assert isinstance(r.text, str)
    assert isinstance(r.source, str)
    assert isinstance(r.label, str)
    db.close()


def test_get_by_id(sample_db):
    path, _, texts = sample_db
    db = nest.open(path)
    for i in [0, 42, 99]:
        r = db.get(i)
        assert r is not None
        assert r.id == i
        assert r.text == texts[i]
    db.close()


def test_get_nonexistent(sample_db):
    path, _, _ = sample_db
    db = nest.open(path)
    assert db.get(9999) is None
    db.close()


def test_context_manager(sample_db):
    path, embeddings, _ = sample_db
    with nest.open(path) as db:
        results = db.search(embeddings[0], k=3)
        assert len(results) == 3


def test_block_compression_consistency(sample_db):
    path, _, texts = sample_db
    db = nest.open(path)
    for i in range(100):
        r = db.get(i)
        assert r.text == texts[i], f"Mismatch at article {i}"
    db.close()


def test_search_different_queries_different_results(sample_db):
    path, embeddings, _ = sample_db
    db = nest.open(path)
    r1 = db.search(embeddings[0], k=5)
    r2 = db.search(embeddings[50], k=5)
    ids1 = {r.id for r in r1}
    ids2 = {r.id for r in r2}
    assert ids1 != ids2
    db.close()
