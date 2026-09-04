"""The notebooks' own per-pair magnitude numbers, parsed from the saved cell outputs.

``magnitudes*.ipynb`` printed, for every similar pair, the signed L1 percent
difference (n1/n2 − 1)·100 computed from vectors the notebook embedded on the fly
(most of those words are not in the cache).  This module parses those printed
values so the full 74-pair record can be compared with the cache-only Phase 2 and
with the GPU re-extraction, using the notebooks' own formula.  The typo pairs
(talk→talkling, index→indicies) are kept but flagged.
"""
from __future__ import annotations

import json
import re

import numpy as np
from scipy import stats

from .common import ROOT, rng
from .load import load
from .pairs import CATEGORIES

# notebook -> (OPT column model, T5 column model)
NOTEBOOKS = {
    "magnitudes.ipynb": ("opt-1.3b", "t5-large"),
    "magnitudes_lrg.ipynb": ("opt-13b", "t5-3b"),
    "magnitudes_xxl.ipynb": (None, "flan-t5-xxl"),   # OPT column is a stale copy-paste, ignored
}
LINE = re.compile(r"^(\S+) -> (\S+)\s+(-?[\d.]+)%(?:\s+(-?[\d.]+)%)?\s*$")   # one column (xxl) or two (OPT, T5)
TYPOS = {"talkling": "talking", "indicies": "indices"}
N_RANDOM = 5000


def _category(a: str, b: str) -> str | None:
    a, b = TYPOS.get(a, a), TYPOS.get(b, b)
    for cat, pairs in CATEGORIES.items():
        if (a, b) in pairs or (b, a) in pairs:
            return cat
    return None


def parse() -> dict[str, list[dict]]:
    """model -> [{a, b, category, pct (signed, notebook formula), typo}]"""
    out: dict[str, list[dict]] = {}
    for nb, (opt, t5) in NOTEBOOKS.items():
        cells = json.load(open(ROOT / nb))["cells"]
        for c in cells:
            if c["cell_type"] != "code":
                continue
            for o in c.get("outputs", []):
                for line in "".join(o.get("text", [])).splitlines():
                    m = LINE.match(line)
                    if not m:
                        continue
                    a, b = m.group(1), m.group(2)
                    cat = _category(a, b)
                    if cat is None:
                        continue
                    rec = {"a": a, "b": b, "category": cat, "typo": a in TYPOS or b in TYPOS}
                    if m.group(4) is None:                       # single T5 column (xxl notebook)
                        out.setdefault(t5, []).append({**rec, "pct": float(m.group(3))})
                    else:
                        if opt:
                            out.setdefault(opt, []).append({**rec, "pct": float(m.group(3))})
                        out.setdefault(t5, []).append({**rec, "pct": float(m.group(4))})
    # de-duplicate (a pair printed twice keeps the first value)
    for model, recs in out.items():
        seen, uniq = set(), []
        for r in recs:
            k = (r["a"], r["b"])
            if k not in seen:
                seen.add(k)
                uniq.append(r)
        out[model] = uniq
    return out


def analyse(include_typos: bool = False) -> dict:
    """|pct| of the notebook pairs, plus a random arm drawn from the CACHE under the same
    L1 ratio formula.  The two arms come from different extractions (on-the-fly vs cached),
    so the p-values here are NOT valid evidence; they are kept in the JSON only so the
    mixed-arm numbers the reviewer's letter was first drafted from remain traceable.
    The report shows the similar-pair arm alone."""
    res = {}
    for model, recs in parse().items():
        vocab, V = load(model)
        l1 = np.abs(V).sum(1)
        g = rng(10)
        N = len(vocab)
        i = g.integers(0, N, N_RANDOM)
        j = (i + g.integers(1, N, N_RANDOM)) % N
        rd = np.abs(l1[i] / l1[j] - 1) * 100
        use = [r for r in recs if include_typos or not r["typo"]]
        d_all = np.array([abs(r["pct"]) for r in use])
        entry = {"n_pairs_parsed": len(recs), "n_used": len(use),
                 "random": {"mean": float(rd.mean()), "median": float(np.median(rd))}, "cats": {}}
        for cat in list(CATEGORIES) + ["all"]:
            d = d_all if cat == "all" else np.array([abs(r["pct"]) for r in use if r["category"] == cat])
            if len(d) == 0:
                continue
            entry["cats"][cat] = {"n": int(len(d)), "mean": float(d.mean()), "median": float(np.median(d)),
                                  "p_less": float(stats.mannwhitneyu(d, rd, alternative="less").pvalue),
                                  "p_greater": float(stats.mannwhitneyu(d, rd, alternative="greater").pvalue)}
        entry["largest"] = sorted(({"pair": f"{r['a']}/{r['b']}", "pct": r["pct"]} for r in use), key=lambda x: -abs(x["pct"]))[:3]
        res[model] = entry
    return res
