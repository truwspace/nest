"""
nest — Nested Embedding Search Tool
A single-file semantic memory. SQLite for embeddings.

    import nest
    db = nest.open("corpus.nest")
    results = db.search("vacina covid", k=10)
    print(results[0].text)

File format: SQLite database with tables:
    articles:  id, text (zstd), source, label
    blobs:     name -> binary data (faiss_index, ordering, manifest)
"""

import io
import json
import sqlite3
import struct
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Union

import faiss
import numpy as np
import zstandard as zstd


__version__ = "0.1.0"
MAGIC = "NEST"

_compressor = zstd.ZstdCompressor(level=9)
_decompressor = zstd.ZstdDecompressor()
_blob_compressor = zstd.ZstdCompressor(level=19)


@dataclass
class SearchResult:
    id: int
    score: float
    text: str
    source: str
    label: str


class NestDB:
    """Read-only interface to a .nest file."""

    def __init__(self, path: str):
        self.path = path
        self.conn = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
        self.conn.row_factory = sqlite3.Row
        self._load_manifest()
        self._load_index()
        self._model = None

    def _load_manifest(self):
        row = self.conn.execute("SELECT data FROM blobs WHERE name='manifest'").fetchone()
        self.manifest = json.loads(row["data"])

    def _load_index(self):
        row = self.conn.execute("SELECT data FROM blobs WHERE name='faiss_index'").fetchone()
        buf = _decompressor.decompress(row["data"])
        self.index = faiss.deserialize_index(np.frombuffer(buf, dtype=np.uint8))
        self._block_cache = {}  # block_id -> list of texts

    def _get_model(self):
        if self._model is None:
            from sentence_transformers import SentenceTransformer
            self._model = SentenceTransformer(self.manifest["embedding_model"])
        return self._model

    def _read_text(self, block_id: int, pos: int) -> str:
        """Read a single text from a block."""
        if block_id not in self._block_cache:
            row = self.conn.execute(
                "SELECT data FROM text_blocks WHERE block_id=?", (block_id,)
            ).fetchone()
            raw = _decompressor.decompress(row["data"])
            n_texts = struct.unpack_from("<I", raw, 0)[0]
            offsets = struct.unpack_from(f"<{n_texts}I", raw, 4)
            header_size = 4 + n_texts * 4
            texts_data = raw[header_size:]
            block_texts = []
            for i in range(n_texts):
                start = offsets[i]
                end = offsets[i + 1] if i + 1 < n_texts else len(texts_data)
                block_texts.append(texts_data[start:end].decode("utf-8"))
            self._block_cache[block_id] = block_texts
            if len(self._block_cache) > 64:
                oldest = next(iter(self._block_cache))
                del self._block_cache[oldest]
        return self._block_cache[block_id][pos]

    def search(self, query: Union[str, np.ndarray], k: int = 10) -> List[SearchResult]:
        """Search by text string or embedding vector."""
        if isinstance(query, str):
            model = self._get_model()
            emb = model.encode([query], normalize_embeddings=True)
            emb = emb.astype(np.float32)
        else:
            emb = np.array(query, dtype=np.float32).reshape(1, -1)

        scores, ids = self.index.search(emb, k)

        results = []
        for score, aid in zip(scores[0], ids[0]):
            if aid < 0:
                continue
            row = self.conn.execute(
                "SELECT block_id, pos_in_block, source, label FROM articles WHERE id=?",
                (int(aid),)
            ).fetchone()
            if row is None:
                continue
            text = self._read_text(row["block_id"], row["pos_in_block"])
            results.append(SearchResult(
                id=int(aid), score=float(score),
                text=text, source=row["source"], label=row["label"],
            ))
        return results

    def get(self, article_id: int) -> Optional[SearchResult]:
        """Get article by ID."""
        row = self.conn.execute(
            "SELECT block_id, pos_in_block, source, label FROM articles WHERE id=?",
            (article_id,)
        ).fetchone()
        if row is None:
            return None
        text = self._read_text(row["block_id"], row["pos_in_block"])
        return SearchResult(id=article_id, score=0.0, text=text,
                            source=row["source"], label=row["label"])

    def neighbors(self, article_id: int, k: int = 10) -> List[SearchResult]:
        """Find semantic neighbors of an article."""
        row = self.conn.execute("SELECT data FROM blobs WHERE name='embeddings'").fetchone()
        if row is None:
            raise ValueError("No raw embeddings stored")
        embs = np.frombuffer(_decompressor.decompress(row["data"]), dtype=np.float16)
        embs = embs.reshape(-1, self.manifest["embedding_dim"]).astype(np.float32)
        emb = embs[article_id:article_id + 1]
        return self.search(emb, k=k + 1)[1:]  # skip self

    @property
    def size(self) -> int:
        return Path(self.path).stat().st_size

    @property
    def n_articles(self) -> int:
        return self.manifest["n_articles"]

    def __repr__(self):
        return f"NestDB({self.path!r}, {self.n_articles} articles, {self.size/1e6:.1f} MB)"

    def close(self):
        self.conn.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()


def open(path: str) -> NestDB:
    """Open a .nest file for reading."""
    return NestDB(path)


def build(
    texts: List[str],
    embeddings: np.ndarray,
    output_path: str,
    sources: Optional[List[str]] = None,
    labels: Optional[List[str]] = None,
    embedding_model: str = "unknown",
    ordering: Optional[np.ndarray] = None,
    store_embeddings: bool = False,
    faiss_nlist: int = 256,
    faiss_m: int = 48,
    faiss_nbits: int = 8,
) -> str:
    """
    Build a .nest file from texts and embeddings.

    Args:
        texts: list of article texts
        embeddings: numpy array (N, D) float32, L2-normalized
        output_path: path to write .nest file
        sources: optional list of source identifiers
        labels: optional list of labels
        embedding_model: name of the embedding model used
        ordering: optional precomputed ordering array
        store_embeddings: if True, store float16 embeddings for exact reconstruction
        faiss_nlist: number of IVF clusters
        faiss_m: number of PQ subquantizers
        faiss_nbits: bits per subquantizer
    """
    N, D = embeddings.shape
    assert len(texts) == N, f"texts ({len(texts)}) != embeddings ({N})"

    if sources is None:
        sources = [""] * N
    if labels is None:
        labels = [""] * N

    t0 = time.time()
    print(f"nest.build: {N} articles, {D} dims")

    # Create SQLite
    path = Path(output_path)
    if path.exists():
        path.unlink()

    conn = sqlite3.connect(str(path))
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=NORMAL")

    conn.execute("""
        CREATE TABLE articles (
            id INTEGER PRIMARY KEY,
            block_id INTEGER NOT NULL,
            pos_in_block INTEGER NOT NULL,
            source TEXT,
            label TEXT
        )
    """)
    conn.execute("""
        CREATE TABLE blobs (
            name TEXT PRIMARY KEY,
            data BLOB NOT NULL
        )
    """)

    # Block-compress texts (256 per block, like GOP for text)
    BLOCK_SIZE = 256
    print(f"  [1/5] Block-compressing texts ({BLOCK_SIZE}/block)...")
    block_comp = zstd.ZstdCompressor(level=19)

    conn.execute("""
        CREATE TABLE text_blocks (
            block_id INTEGER PRIMARY KEY,
            data BLOB NOT NULL
        )
    """)

    n_blocks = (N + BLOCK_SIZE - 1) // BLOCK_SIZE
    for b in range(n_blocks):
        start = b * BLOCK_SIZE
        end = min(start + BLOCK_SIZE, N)
        # Pack: [n_texts(u32)] + [offset0(u32), offset1(u32), ...] + [text0, text1, ...]
        encoded_texts = [texts[i].encode("utf-8") for i in range(start, end)]
        n_in_block = len(encoded_texts)
        offsets = []
        pos = 0
        for et in encoded_texts:
            offsets.append(pos)
            pos += len(et)
        header = struct.pack(f"<I{n_in_block}I", n_in_block, *offsets)
        raw_block = header + b"".join(encoded_texts)
        compressed = block_comp.compress(raw_block)
        conn.execute("INSERT INTO text_blocks VALUES (?,?)", (b, compressed))

    # Insert article metadata (no text blob, just block reference)
    print("  [2/5] Inserting article metadata...")
    batch = []
    for i in range(N):
        batch.append((i, i // BLOCK_SIZE, i % BLOCK_SIZE, sources[i], labels[i]))
        if len(batch) >= 1000:
            conn.executemany("INSERT INTO articles VALUES (?,?,?,?,?)", batch)
            batch = []
    if batch:
        conn.executemany("INSERT INTO articles VALUES (?,?,?,?,?)", batch)
    conn.commit()

    # Build FAISS index
    embs_f32 = embeddings.astype(np.float32)
    if N < 1000:
        print("  [3/5] Building FAISS FlatIP index (small corpus)...")
        index = faiss.IndexFlatIP(D)
        index.add(embs_f32)
        nlist = 0
    else:
        print("  [3/5] Building FAISS IVF+PQ index...")
        nlist = min(faiss_nlist, N // 40)
        quantizer = faiss.IndexFlatIP(D)
        index = faiss.IndexIVFPQ(quantizer, D, nlist, faiss_m, faiss_nbits)
        index.train(embs_f32)
        index.add(embs_f32)
        index.nprobe = min(16, nlist)

    # Serialize FAISS
    buf = faiss.serialize_index(index)
    faiss_compressed = _blob_compressor.compress(bytes(buf))
    conn.execute("INSERT INTO blobs VALUES (?,?)", ("faiss_index", faiss_compressed))
    print(f"    FAISS: {len(faiss_compressed):,} bytes")

    # Ordering
    if ordering is not None:
        print("  [4/5] Storing ordering...")
        order_bytes = _blob_compressor.compress(ordering.astype(np.int32).tobytes())
        conn.execute("INSERT INTO blobs VALUES (?,?)", ("ordering", order_bytes))
        print(f"    Ordering: {len(order_bytes):,} bytes")
    else:
        print("  [4/5] No ordering provided, skipping")

    # Optional: store float16 embeddings
    if store_embeddings:
        print("  [4b] Storing float16 embeddings...")
        emb_bytes = _blob_compressor.compress(embeddings.astype(np.float16).tobytes())
        conn.execute("INSERT INTO blobs VALUES (?,?)", ("embeddings", emb_bytes))
        print(f"    Embeddings: {len(emb_bytes):,} bytes")

    # Manifest
    manifest = {
        "magic": MAGIC,
        "version": __version__,
        "n_articles": N,
        "embedding_dim": D,
        "embedding_model": embedding_model,
        "faiss_nlist": nlist,
        "faiss_m": faiss_m,
        "faiss_nbits": faiss_nbits,
        "created": time.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "has_ordering": ordering is not None,
        "has_embeddings": store_embeddings,
    }
    conn.execute("INSERT INTO blobs VALUES (?,?)",
                 ("manifest", json.dumps(manifest).encode("utf-8")))

    conn.commit()
    conn.execute("VACUUM")
    conn.close()

    elapsed = time.time() - t0
    size = path.stat().st_size
    print(f"  [5/5] Done: {path} ({size/1e6:.1f} MB, {elapsed:.1f}s)")
    return str(path)
