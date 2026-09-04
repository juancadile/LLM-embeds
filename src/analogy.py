"""Mikolov-style analogies, with the original bug fixed.

In ``negatives*.ipynb`` the wrapper ``mikolov(start, less, more, target)`` called
``negative(start, less, more, model)``: the model name landed in the ``end`` slot, so
every reported score was cos(A − B + C, v("gpt"|"opt"|"t5")) computed with the default
GPT model.  Here: score = cos(v(start) − v(less) + v(more), v(target)), and the rank of
*target* among vocab words excluding the three inputs.
"""
from __future__ import annotations

import numpy as np

from .similarity import Space

ANALOGIES = [
    ("king", "man", "woman", "queen"),
    ("man", "king", "queen", "woman"),
    ("husband", "man", "woman", "wife"),
    ("brother", "male", "female", "sister"),
    ("father", "man", "woman", "mother"),
    ("kitten", "cat", "dog", "puppy"),
    ("bigger", "big", "small", "smaller"),
    ("walked", "walk", "run", "ran"),
]


def run_analogies(space: Space, k: int = 5):
    rows = []
    for a, b, c, d in ANALOGIES:
        if not all(w in space for w in (a, b, c, d)):
            rows.append({"analogy": f"{a} - {b} + {c} -> {d}", "skipped": "word not in vocab"})
            continue
        q = space.vec(a) - space.vec(b) + space.vec(c)
        rank, cos = space.rank_of(q, d, exclude=(a, b, c))
        top = space.most_similar(q, k, exclude=(a, b, c))
        rows.append({
            "analogy": f"{a} - {b} + {c} -> {d}",
            "rank": rank, "cos": cos,
            "top": top,
            "benchmark_cos_a_d": float(space.Vn[space.idx[a]] @ space.Vn[space.idx[d]]),
        })
    return rows
