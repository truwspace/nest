"""Embed a single text query using the same sentence-transformers model
the corpus was built with.

Invoked by the Rust CLI's `nest search-text` subcommand. Stays in
Python because (a) sentence-transformers is the same toolchain used at
build time, so vectors are bit-identical (modulo float ops), and (b)
keeping the CLI binary lean — no candle/onnxruntime dependency.

Stdin/argv contract:

  python3 embed_query.py <model_name> <query>     -> writes JSON array of f32 to stdout
  python3 embed_query.py --embed-dim <model_name> -> writes the model's embedding dim and exits

Stays L2-normalized so the runtime's cosine assumption holds. Errors go
to stderr with a non-zero exit code; the CLI surfaces them.
"""
from __future__ import annotations

import argparse
import json
import math
import sys


def _embed(model_name: str, query: str) -> list[float]:
    from sentence_transformers import SentenceTransformer  # local import: heavy

    model = SentenceTransformer(model_name)
    vec = model.encode(
        [query], normalize_embeddings=True, convert_to_numpy=True
    )[0]
    return [float(x) for x in vec]


def _embed_dim(model_name: str) -> int:
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(model_name)
    return int(model.get_sentence_embedding_dimension())


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--embed-dim", action="store_true",
                   help="print the model's embedding dim and exit")
    p.add_argument("model")
    p.add_argument("query", nargs="?", default="")
    args = p.parse_args()

    if args.embed_dim:
        print(_embed_dim(args.model))
        return 0

    if not args.query:
        print("error: query required", file=sys.stderr)
        return 2

    vec = _embed(args.model, args.query)
    # Defensive re-normalize in case the model didn't.
    n = math.sqrt(sum(x * x for x in vec))
    if n > 0:
        vec = [x / n for x in vec]
    json.dump(vec, sys.stdout)
    return 0


if __name__ == "__main__":
    sys.exit(main())
