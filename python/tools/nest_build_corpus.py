"""
nest_build_corpus.py — build a deterministic .nest from the seven PT-BR
fake-news datasets under `database/`.

Steps:
  1. read each source
  2. normalize to (text, label, source, title, url)
  3. filter empty/short text (len > 20)
  4. dedupe by sha256(text), keep first
  5. embed each row (sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2)
  6. call builder.Pipeline (existing tool) to chunk + cache + emit
  7. write data/<name>.nest
  8. shell out to `nest validate`
  9. print report: original counts vs post-dedup vs in .nest

Run:
  python3 python/tools/nest_build_corpus.py
  python3 python/tools/nest_build_corpus.py --out data/foo.nest
"""

from __future__ import annotations

import argparse
import hashlib
import os
import re
import subprocess
import sys
from collections import Counter
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "python"))
from builder import BuildConfig, ChunkSpec, Pipeline  # noqa: E402

DB = REPO / "database"
DATA = REPO / "data"
NEST_BIN = REPO / "target" / "release" / "nest"

EMBED_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
EMBED_DIM = 384
MIN_TEXT_LEN = 20


def _ws(s) -> str:
    return re.sub(r"\s+", " ", s).strip() if isinstance(s, str) else ""


def _row(text, label, source, title="", url=""):
    return {
        "text": _ws(text),
        "label": label,
        "source": source,
        "title": _ws(title),
        "url": _ws(url),
    }


# ── loaders ────────────────────────────────────────────────────────────────


def load_fakebr_hf() -> pd.DataFrame:
    rows = []
    for split in ("train.csv", "test.csv"):
        df = pd.read_csv(DB / "FakeBr-hf" / split)
        df["label"] = df["label"].astype(str).str.strip()
        rows += [_row(r["text"], r["label"], "FakeBr-hf") for _, r in df.iterrows()]
    return pd.DataFrame(rows)


def load_faketrue_br_hf() -> pd.DataFrame:
    rows = []
    for split in ("train.csv", "test.csv"):
        df = pd.read_csv(DB / "FakeTrue.Br-hf" / split)
        df["label"] = df["label"].astype(str).str.strip()
        rows += [_row(r["text"], r["label"], "FakeTrue.Br-hf") for _, r in df.iterrows()]
    return pd.DataFrame(rows)


def load_fake_br_corpus() -> pd.DataFrame:
    df = pd.read_csv(DB / "Fake.br-Corpus" / "preprocessed" / "pre-processed.csv")
    m = {"fake": "False", "true": "True"}
    rows = [
        _row(
            r["preprocessed_news"],
            m.get(str(r["label"]).lower(), str(r["label"])),
            "Fake.br-Corpus",
        )
        for _, r in df.iterrows()
    ]
    return pd.DataFrame(rows)


def load_fake_recogna() -> pd.DataFrame:
    df = pd.read_csv(DB / "FakeRecogna" / "dataset" / "FakeRecogna.csv")
    m = {0: "False", 1: "True", "0": "False", "1": "True"}
    rows = []
    for _, r in df.iterrows():
        body = " ".join(
            filter(
                None,
                [
                    _ws(r.get("Titulo", "")),
                    _ws(r.get("Subtitulo", "")),
                    _ws(r.get("Noticia", "")),
                ],
            )
        )
        rows.append(
            _row(
                body,
                m.get(r["Classe"], str(r["Classe"])),
                "FakeRecogna",
                title=r.get("Titulo", ""),
                url=r.get("URL", ""),
            )
        )
    return pd.DataFrame(rows)


def load_faketrue_br() -> pd.DataFrame:
    df = pd.read_csv(DB / "FakeTrue.Br" / "FakeTrueBr_corpus.csv")
    rows = []
    for _, r in df.iterrows():
        rows.append(
            _row(
                r.get("fake", ""),
                "False",
                "FakeTrue.Br",
                title=r.get("title_fake", ""),
                url=r.get("link_f", ""),
            )
        )
        rows.append(_row(r.get("true", ""), "True", "FakeTrue.Br", url=r.get("link_t", "")))
    return pd.DataFrame(rows)


def load_factck_br() -> pd.DataFrame:
    df = pd.read_csv(DB / "factck-br" / "FACTCKBR.tsv", sep="\t")
    rows = []
    for _, r in df.iterrows():
        rating = int(r["ratingValue"]) if pd.notna(r["ratingValue"]) else 3
        if rating == 3:
            continue  # ambiguous
        body = " ".join(
            filter(
                None,
                [
                    _ws(r.get("claimReviewed", "")),
                    _ws(r.get("reviewBody", "")),
                ],
            )
        )
        rows.append(
            _row(
                body,
                "False" if rating <= 2 else "True",
                "factck-br",
                title=r.get("title", ""),
                url=r.get("URL", ""),
            )
        )
    return pd.DataFrame(rows)


def load_bilstm() -> pd.DataFrame:
    df = pd.read_parquet(
        DB / "portuguese-fake-news-classifier-bilstm-combined" / "corpus" / "corpus_test_df.parquet"
    )

    def lbl(v):
        if isinstance(v, bool):
            return "True" if v else "False"
        return str(v).strip()

    rows = [
        _row(r["text"], lbl(r["label"]), "portuguese-fake-news-classifier-bilstm-combined")
        for _, r in df.iterrows()
    ]
    return pd.DataFrame(rows)


SOURCES = [
    ("FakeBr-hf", load_fakebr_hf),
    ("FakeTrue.Br-hf", load_faketrue_br_hf),
    ("Fake.br-Corpus", load_fake_br_corpus),
    ("FakeRecogna", load_fake_recogna),
    ("FakeTrue.Br", load_faketrue_br),
    ("factck-br", load_factck_br),
    ("portuguese-fake-news-classifier-bilstm-combined", load_bilstm),
]


# ── main ───────────────────────────────────────────────────────────────────


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--out",
        default=str(DATA / "corpus_next.v1.nest"),
        help="output .nest path (default: data/corpus_next.v1.nest)",
    )
    ap.add_argument(
        "--cache",
        default=str(DB / "corpus-next" / "embed_cache.sqlite"),
        help="sqlite scratch path for embedding cache",
    )
    args = ap.parse_args()

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    Path(args.cache).parent.mkdir(parents=True, exist_ok=True)

    # 1. read
    print("→ reading sources")
    raw = {}
    for name, loader in SOURCES:
        df = loader()
        raw[name] = df
        print(f"   {name:50s} {len(df):>7,} rows")

    # 2. normalize already done in loaders; concatenate
    combined = pd.concat([raw[n] for n, _ in SOURCES], ignore_index=True)
    n_before = len(combined)

    # 3. filter empty/short text
    combined = combined[combined["text"].str.len() > MIN_TEXT_LEN].copy()
    n_after_filter = len(combined)

    # 4. dedupe by sha256(text), keep first occurrence (priority = listing order)
    combined["_h"] = combined["text"].apply(
        lambda t: hashlib.sha256(t.encode("utf-8", errors="replace")).hexdigest()
    )
    combined = combined.drop_duplicates(subset="_h", keep="first").drop(columns="_h")
    n_after_dedup = len(combined)
    print(
        f"   raw rows={n_before:,}  after-filter={n_after_filter:,}  after-dedup={n_after_dedup:,}"
    )

    # 5+6. embed and emit via Pipeline
    print(f"\n→ loading {EMBED_MODEL}")
    from sentence_transformers import SentenceTransformer  # local import

    model = SentenceTransformer(EMBED_MODEL)
    print(f"   device: {model.device}")

    def embedder(specs):
        embs = model.encode(
            [s.canonical_text for s in specs],
            batch_size=64,
            normalize_embeddings=True,
            show_progress_bar=True,
            convert_to_numpy=True,
        )
        return [e.astype("float32").tolist() for e in embs]

    specs, sources, labels, titles, urls = [], [], [], [], []
    for _, r in combined.iterrows():
        ub = r["text"].encode("utf-8")
        h = hashlib.sha256(ub).hexdigest()[:16]
        specs.append(
            ChunkSpec(
                canonical_text=r["text"],
                source_uri=f"corpus-next://{r['source']}/{h}",
                byte_start=0,
                byte_end=len(ub),
            )
        )
        sources.append(r["source"])
        labels.append(r["label"])
        titles.append(r.get("title", "") or "")
        urls.append(r.get("url", "") or "")

    # Compute a real, reproducible model fingerprint instead of the
    # legacy zero-placeholder. This makes `nest search-text` able to
    # detect if the user feeds a query embedded by a different model.
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from model_fingerprint import (  # noqa: E402
        compute_model_fingerprint,
        fingerprint_to_model_hash,
        resolve_model_dir,
    )

    model_dir = resolve_model_dir(EMBED_MODEL)
    fp = compute_model_fingerprint(model_dir, model_id=EMBED_MODEL)
    real_model_hash = fingerprint_to_model_hash(fp)
    print(f"model fingerprint: {fp.to_dict()}")
    print(f"model_hash: {real_model_hash}")

    cfg = BuildConfig(
        output_path=str(out_path),
        embedding_model=EMBED_MODEL,
        embedding_dim=EMBED_DIM,
        chunker_version="corpus-next/v0.1.0",
        model_hash=real_model_hash,
        title="corpus-next",
        version="v0.1.0",
        description=(
            "Unified PT-BR fake-news corpus combining 7 public datasets, "
            "deduplicated by SHA-256 of text."
        ),
        license="mixed (per-source)",
        reproducible=True,
    )
    pipe = Pipeline(cfg, embedder=embedder, scratch_db=args.cache)
    pipe.add_many(specs)
    pipe.emit(
        provenance={
            "legacy_source": "corpus-next (unified PT-BR fake-news corpus)",
            "legacy_version": "v0.1.0",
            "datasets": [n for n, _ in SOURCES],
            "labels": labels,
            "sources": sources,
            "titles": titles,
            "urls": urls,
            "note": "row index matches the post-dedup corpus",
        }
    )
    pipe.close()
    size = out_path.stat().st_size
    print(f"\n✓ .nest → {out_path}  ({size / 1e6:.2f} MB)")

    # 7. validate via the released CLI
    print("\n→ nest validate")
    if NEST_BIN.exists():
        subprocess.run([str(NEST_BIN), "validate", str(out_path)], check=True)
    else:
        print(f"   (skipped: {NEST_BIN} not found — run `cargo build --release -p nest-cli`)")

    # 8. report
    in_nest = Counter(sources)
    print("\n" + "=" * 90)
    print(f"{'source':50s} {'original':>10s} {'in .nest':>10s} {'∆':>10s}")
    print("=" * 90)
    total_orig = total_nest = 0
    for name, _ in SOURCES:
        orig = len(raw[name])
        nest_n = in_nest.get(name, 0)
        delta = nest_n - orig
        total_orig += orig
        total_nest += nest_n
        print(f"{name:50s} {orig:>10,} {nest_n:>10,} {delta:>+10,}")
    print("-" * 90)
    print(f"{'TOTAL':50s} {total_orig:>10,} {total_nest:>10,} {total_nest - total_orig:>+10,}")
    print(f"\nlabels in .nest: {dict(Counter(labels))}")


if __name__ == "__main__":
    main()
