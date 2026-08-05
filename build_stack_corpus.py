#!/usr/bin/env python3
"""Build corpus.bin: shuffled, ASCII-filtered Lisp-family code from The Stack.

Run on the training box (needs `pip install datasets` and an HF token with
access to bigcode/the-stack-dedup, via the HF_TOKEN env var):

    HF_TOKEN=... python3 build_stack_corpus.py --max-gb 5 --out corpus.bin
"""

import argparse
import random

LANGS = ["common-lisp", "scheme", "emacs-lisp", "racket", "clojure"]
KEEP = frozenset({9, 10} | set(range(32, 127)))
MIN_DOC_BYTES = 100
MLX_MAX_ELEMENTS = 2**31 - 1  # MLX arrays use int32 for shape dimensions


def ascii_filter(text: str) -> bytes:
    return bytes(b for b in text.encode("utf-8", "ignore") if b in KEEP)


def shuffle_docs(docs: list, seed: int) -> list:
    random.Random(seed).shuffle(docs)
    return docs


def byte_budget(max_gb: float) -> int:
    return min(int(max_gb * 1e9), MLX_MAX_ELEMENTS)


def collect(max_bytes: int):
    from datasets import load_dataset
    docs, total = [], 0
    per_lang = max_bytes // len(LANGS)
    for lang in LANGS:
        got = 0
        ds = load_dataset("bigcode/the-stack-dedup",
                          data_dir=f"data/{lang}", split="train",
                          streaming=True)
        for row in ds:
            b = ascii_filter(row["content"])
            if len(b) < MIN_DOC_BYTES:
                continue
            docs.append(b + b"\n\n")
            got += len(b) + 2
            if got >= per_lang:
                break
        print(f"{lang}: {got / 1e6:.1f} MB", flush=True)
        total += got
    return docs, total


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-gb", type=float, default=5.0)
    ap.add_argument("--out", default="corpus.bin")
    ap.add_argument("--seed", type=int, default=1234)
    args = ap.parse_args()
    max_bytes = byte_budget(args.max_gb)
    if max_bytes < args.max_gb * 1e9:
        print(f"clamping corpus to MLX int32 limit: {max_bytes / 1e9:.2f} GB (requested {args.max_gb} GB)", flush=True)
    docs, total = collect(max_bytes)
    shuffle_docs(docs, args.seed)
    with open(args.out, "wb") as f:
        for d in docs:
            f.write(d)
    print(f"wrote {args.out}: {total / 1e9:.2f} GB in {len(docs)} docs (shuffled)")


if __name__ == "__main__":
    main()
