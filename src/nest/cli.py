"""nest CLI — build and search .nest files."""

import argparse
import csv
import json
import sys
import time
from pathlib import Path

import numpy as np


def cmd_build(args):
    """Build a .nest file from CSV + embeddings."""
    from nest.core import build

    csv_path = Path(args.input)
    if not csv_path.exists():
        print(f"Error: {csv_path} not found", file=sys.stderr)
        sys.exit(1)

    texts, sources, labels = [], [], []
    with open(csv_path, encoding="utf-8", errors="replace") as f:
        reader = csv.DictReader(f)
        for row in reader:
            texts.append(row.get("text", ""))
            sources.append(row.get("source", ""))
            labels.append(row.get("label", ""))

    if args.embeddings:
        emb_path = Path(args.embeddings)
        if emb_path.suffix == ".npy":
            embeddings = np.load(emb_path).astype(np.float32)
        else:
            print(f"Error: unsupported embedding format {emb_path.suffix}", file=sys.stderr)
            sys.exit(1)
    else:
        print("Computing embeddings (requires sentence-transformers)...")
        from sentence_transformers import SentenceTransformer
        model_name = args.model or "paraphrase-multilingual-MiniLM-L12-v2"
        model = SentenceTransformer(model_name)
        embeddings = model.encode(texts, normalize_embeddings=True, show_progress_bar=True)
        embeddings = embeddings.astype(np.float32)

    N = min(len(texts), embeddings.shape[0])
    texts, sources, labels = texts[:N], sources[:N], labels[:N]
    embeddings = embeddings[:N]

    output = args.output or csv_path.with_suffix(".nest")
    build(
        texts=texts,
        embeddings=embeddings,
        output_path=str(output),
        sources=sources,
        labels=labels,
        embedding_model=args.model or "paraphrase-multilingual-MiniLM-L12-v2",
    )


def cmd_search(args):
    """Search a .nest file."""
    from nest.core import open as nest_open

    db = nest_open(args.db)
    results = db.search(args.query, k=args.k)

    for i, r in enumerate(results):
        print(f"\n[{i+1}] score={r.score:.4f} id={r.id} src={r.source} label={r.label}")
        print(f"    {r.text[:200]}")

    db.close()


def cmd_info(args):
    """Show info about a .nest file."""
    from nest.core import open as nest_open

    db = nest_open(args.db)
    print(db)
    print(json.dumps(db.manifest, indent=2))
    db.close()


def cmd_get(args):
    """Get article by ID."""
    from nest.core import open as nest_open

    db = nest_open(args.db)
    article = db.get(args.id)
    if article is None:
        print(f"Article {args.id} not found", file=sys.stderr)
        sys.exit(1)
    print(f"id={article.id} source={article.source} label={article.label}")
    print(article.text)
    db.close()


def main():
    parser = argparse.ArgumentParser(prog="nest", description="Nested Embedding Search Tool")
    sub = parser.add_subparsers(dest="command")

    # build
    p_build = sub.add_parser("build", help="Build .nest from CSV + embeddings")
    p_build.add_argument("input", help="CSV file with 'text' column (optional: 'source', 'label')")
    p_build.add_argument("-e", "--embeddings", help="Path to .npy embeddings (skip if using --model)")
    p_build.add_argument("-o", "--output", help="Output .nest path")
    p_build.add_argument("-m", "--model", help="Sentence-transformer model name", default=None)

    # search
    p_search = sub.add_parser("search", help="Search a .nest file")
    p_search.add_argument("query", help="Search query (text)")
    p_search.add_argument("-d", "--db", required=True, help="Path to .nest file")
    p_search.add_argument("-k", type=int, default=10, help="Number of results")

    # info
    p_info = sub.add_parser("info", help="Show .nest file info")
    p_info.add_argument("db", help="Path to .nest file")

    # get
    p_get = sub.add_parser("get", help="Get article by ID")
    p_get.add_argument("id", type=int, help="Article ID")
    p_get.add_argument("-d", "--db", required=True, help="Path to .nest file")

    args = parser.parse_args()
    if args.command == "build":
        cmd_build(args)
    elif args.command == "search":
        cmd_search(args)
    elif args.command == "info":
        cmd_info(args)
    elif args.command == "get":
        cmd_get(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
