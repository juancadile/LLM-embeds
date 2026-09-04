"""Cognate filter for define2: exclude the target, its plural/singular, words sharing a
prefix with it, and words within Levenshtein distance 2.

The original notebooks used a 3-letter prefix test against the target plus an
``inflect`` plural list; the paper reports results from two different filter settings.
Here one filter is used everywhere.
"""
from __future__ import annotations

import inflect
from rapidfuzz.distance import Levenshtein

PREFIX_LEN = 3
MAX_EDIT = 2

_inf = inflect.engine()


def inflections(word: str) -> set[str]:
    out = {word}
    p = _inf.plural(word)
    if p:
        out.add(p)
    s = _inf.singular_noun(word)
    if s:
        out.add(s)
    return out


def is_cognate(word: str, target: str, prefix_len: int = PREFIX_LEN, max_edit: int = MAX_EDIT) -> bool:
    if word == target:
        return True
    if word in inflections(target) or target in inflections(word):
        return True
    if len(target) >= prefix_len and word[:prefix_len] == target[:prefix_len]:
        return True
    if Levenshtein.distance(word, target) <= max_edit:
        return True
    return False


def cognate_indices(target: str, vocab: list[str], **kw) -> list[int]:
    return [i for i, w in enumerate(vocab) if is_cognate(w, target, **kw)]
