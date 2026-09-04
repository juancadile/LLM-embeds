"""Word-pair lists for the magnitude study, recovered from magnitudes*.ipynb.

Typos in the notebooks are fixed here: ``talkling`` -> ``talking`` and
``indicies`` -> ``indices`` (the small-model notebook already had ``indices``).
Groups with more than two forms expand to all unordered pairs, as the notebooks did.
"""
from __future__ import annotations

from itertools import combinations

US_UK = [
    ("color", "colour"), ("flavor", "flavour"), ("apologize", "apologise"),
    ("organize", "organise"), ("analyze", "analyse"), ("traveled", "travelled"),
    ("maneuver", "manoeuvre"), ("pediatric", "paediatric"), ("license", "licence"),
    ("offence", "offense"), ("analog", "analogue"),
]

PLURAL_GROUPS = [
    ("wife", "wives"), ("dog", "dogs"), ("tooth", "teeth"), ("man", "men"),
    ("potato", "potatoes"), ("mouse", "mice"), ("child", "children"),
    ("person", "persons", "people"), ("radius", "radii"), ("nucleus", "nuclei"),
    ("alumna", "alumnae", "alumnus", "alumni"), ("crisis", "crises"),
    ("thesis", "theses"), ("phenomenon", "phenomena"), ("datum", "data"),
    ("bacterium", "bacteria"), ("index", "indices"),
    ("appendix", "appendices", "appendixes"),
]

VERB_GROUPS = [
    ("chew", "chews", "chewing", "chewed"),
    ("talk", "talks", "talking", "talked"),
    ("run", "runs", "running", "ran"),
    ("think", "thinks", "thinking", "thought"),
    ("drive", "drives", "driving", "drove"),
    ("cry", "cries", "crying", "cried"),
]


def _expand(groups):
    out = []
    for g in groups:
        out.extend(combinations(g, 2))
    return out


CATEGORIES: dict[str, list[tuple[str, str]]] = {
    "us_uk": list(US_UK),
    "plural": _expand(PLURAL_GROUPS),
    "verb": _expand(VERB_GROUPS),
}

ALL_WORDS = sorted({w for pairs in CATEGORIES.values() for p in pairs for w in p})
