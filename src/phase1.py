"""Phase 1: reproduce the paper's define(a+b) table, add the missing controls, rerun
define2 with one filter on all cached models, and fix the analogy test."""
from __future__ import annotations

import time

import numpy as np

from .analogy import run_analogies
from .common import RESULTS, dump_json, fmt, md_table, rng
from .filters import cognate_indices
from .load import DEFINE_MODELS, load
from .similarity import Space

# (a, b, target) from paper p. 16, plus the bachelor case (footnote 22)
P16 = [
    ("young", "dog", "puppy"), ("young", "cat", "kitten"), ("young", "duck", "duckling"),
    ("female", "spouse", "wife"), ("male", "spouse", "husband"),
    ("male", "sibling", "brother"), ("female", "sibling", "sister"),
]
BACHELOR = ("man", "unmarried", "bachelor")
PAPER_P16 = {  # target cosine as printed in the paper
    "puppy": 0.90453, "kitten": 0.899662, "duckling": 0.90637, "wife": 0.93895,
    "husband": 0.93025, "brother": 0.92325, "sister": 0.93183,
}
PAPER_P16_AB = {  # cosine of a (and b) to a+b as printed
    "puppy": 0.94225, "kitten": 0.93312, "duckling": 0.93372, "wife": 0.95728,
    "husband": 0.95423, "brother": 0.95153, "sister": 0.95066,
}

# define2 targets from pp. 18-19 with the paper's reported (pair, cosine)
P18_19 = {
    "wife": (("spouse", "woman"), 0.94900), "duckling": (("duck", "youngster"), 0.92606),
    "puppy": (("dog", "kitten"), 0.93234), "foal": (("horse", "puppy"), 0.90931),
    "freedom": (("independence", "liberty"), 0.94604),
    "autonomy": (("control", "independence"), 0.91767),
    "justice": (("fairness", "judicial"), 0.91696),
    "knowledge": (("information", "wisdom"), 0.93445),
    "rationality": (("logical", "reasoning"), 0.923637),
    "causation": (("consequence", "correlation"), 0.91046),
}

N_RANDOM_PAIRS = 1000
N_NULL = 100
MODES = ("raw", "center")


def _spaces(name, modes=MODES):
    vocab, V = load(name)
    return {m: Space.from_raw(vocab, V, name, m) for m in modes}


# ---------------------------------------------------------------------------
def reproduce_p16(S: Space):
    rows = []
    for a, b, t in P16:
        top = S.define_ab(a, b, 5)
        d = dict(top)
        rows.append({"a": a, "b": b, "target": t, "top5": top,
                     "cos_target": d.get(t), "cos_a": d.get(a), "cos_b": d.get(b),
                     "paper_target": PAPER_P16[t], "paper_ab": PAPER_P16_AB[t]})
    return rows


def controls(S: Space, triples):
    rows = []
    for a, b, t in triples:
        r = {"a": a, "b": b, "target": t}
        for label, q in (("a", S.vec(a)), ("b", S.vec(b)), ("a+b", S.vec(a) + S.vec(b))):
            rank, cos = S.rank_of(q, t, exclude=(a, b))
            r[f"rank_{label}"], r[f"cos_{label}"] = rank, cos
        rows.append(r)
    return rows


def random_pair_baseline(S: Space, n=N_RANDOM_PAIRS, seed_offset=0):
    g = rng(seed_offset)
    N = len(S.vocab)
    best = np.empty(n, dtype=np.float32)
    for k in range(n):
        i, j = g.choice(N, 2, replace=False)
        s = S.cosines(S.V[i] + S.V[j])
        s[[i, j]] = -np.inf
        best[k] = s.max()
    return {"n": n, "mean": float(best.mean()), "p10": float(np.percentile(best, 10)),
            "p50": float(np.percentile(best, 50)), "p90": float(np.percentile(best, 90))}


def define2_all(S: Space, k=3):
    out = {}
    for t in P18_19:
        if t not in S:
            out[t] = None
            continue
        ex = cognate_indices(t, S.vocab)
        out[t] = {"top": S.define2_word(t, k, ex), "n_excluded": len(ex)}
    return out


def define2_null(S: Space, n=N_NULL, seed_offset=1):
    """Best define2 score for n random unit-vector targets and for n random word targets."""
    g = rng(seed_offset)
    d = S.V.shape[1]
    unit_best, word_best = [], []
    for _ in range(n):
        q = g.standard_normal(d).astype(np.float32)
        unit_best.append(S.define2(q, 1)[0][2])
    N = len(S.vocab)
    for i in g.choice(N, n, replace=False):
        w = S.vocab[i]
        ex = cognate_indices(w, S.vocab)
        word_best.append(S.define2_word(w, 1, ex)[0][2])
    def summ(x):
        x = np.array(x)
        return {"mean": float(x.mean()), "p50": float(np.median(x)),
                "p90": float(np.percentile(x, 90)), "max": float(x.max())}
    return {"random_unit": summ(unit_best), "random_word": summ(word_best)}


# ---------------------------------------------------------------------------
def run(models=DEFINE_MODELS):
    t0 = time.time()
    res = {"models": {}, "controls_model": "curie"}

    # 3 + 4: Curie reproduction, controls and baseline (raw and centred)
    cur = _spaces("curie")
    res["p16"] = reproduce_p16(cur["raw"])
    res["controls"] = {m: controls(cur[m], P16 + [BACHELOR]) for m in MODES}
    res["baseline"] = {m: random_pair_baseline(cur[m]) for m in MODES}

    # 5 + 6: define2, null and analogies on every model
    for name in models:
        sp = cur if name == "curie" else _spaces(name)
        res["models"][name] = {
            m: {"define2": define2_all(sp[m]), "null": define2_null(sp[m]),
                "analogies": run_analogies(sp[m])}
            for m in MODES
        }
        print(f"[phase1] {name} done ({time.time() - t0:.0f}s)", flush=True)
    res["seconds"] = time.time() - t0
    dump_json(res, RESULTS / "phase1.json")
    return res


# ---------------------------------------------------------------------------
def report(res) -> str:
    L = ["## Phase 1 — define(a+b), controls, define2, analogies", ""]

    # p.16 reproduction
    L += ["### 1.1 Reproduction of the p. 16 table (GPT-3 Curie, raw)", ""]
    rows, ok = [], True
    for r in res["p16"]:
        match = abs(r["cos_target"] - r["paper_target"]) < 5e-5
        ok &= match
        rows.append([f"{r['a']} + {r['b']} → {r['target']}", fmt(r["cos_a"]), fmt(r["paper_ab"]),
                     fmt(r["cos_target"]), fmt(r["paper_target"]), "✓" if match else "✗",
                     ", ".join(f"{w} {fmt(c, 3)}" for w, c in r["top5"][2:])])
    L.append(md_table(["a + b → target", "cos(a, a+b)", "paper", "cos(target, a+b)", "paper", "match", "next three"], rows))
    L += ["", f"**Reading.** {'All seven cosines reproduce to four decimals' if ok else 'At least one value does not reproduce'}, "
          "so the cached Curie file and this loader are the same data the paper used. "
          "cos(a, a+b) equals cos(b, a+b) exactly because Curie vectors are unit length, so a and b heading the list is arithmetic, not a finding.", ""]

    # controls
    for m in MODES:
        L += [f"### 1.2 Controls, Curie, {m}", ""]
        rows = []
        for r in res["controls"][m]:
            rows.append([f"{r['a']} + {r['b']} → {r['target']}", r["rank_a"], fmt(r["cos_a"]),
                         r["rank_b"], fmt(r["cos_b"]), r["rank_a+b"], fmt(r["cos_a+b"])])
        L.append(md_table(["a + b → target", "rank | v(a)", "cos", "rank | v(b)", "cos", "rank | v(a)+v(b)", "cos"], rows))
        b = res["baseline"][m]
        L += ["", f"Random-pair baseline ({b['n']} seeded pairs, best non-a/b neighbour of v(a)+v(b)): "
              f"mean {fmt(b['mean'])}, p10 {fmt(b['p10'])}, p50 {fmt(b['p50'])}, p90 {fmt(b['p90'])}.", ""]
    c_raw = {r["target"]: r for r in res["controls"]["raw"]}
    animals = [t for t in ("puppy", "kitten", "duckling") if c_raw[t]["rank_b"] == 1]
    L += ["**Reading.** Ranks exclude a and b themselves. " +
          (f"For {', '.join(animals)} the target is already the nearest neighbour of v(b) alone; adding v(a) changes the cosine by a few thousandths and the rank not at all. " if animals else "") +
          "The kinship cases move the target from rank 2–3 under v(b) to rank 1 under the sum, a real but small effect. "
          f"The bachelor example stays at rank {c_raw['bachelor']['rank_a+b']}. "
          "The baseline shows what a random sum scores against its best neighbour; a p. 16 cosine near the baseline mean is typical, not evidence of composition. "
          "Mean-centering removes the common component that inflates every raw cosine; compare the two tables to see which effects survive.", ""]

    # define2
    L += ["### 1.3 define2 with one cognate filter (plural + 3-letter prefix + edit distance ≤ 2), top-3 per target", ""]
    for m in MODES:
        L += [f"**{m}**", ""]
        header = ["target", "paper (Curie)"] + list(res["models"].keys())
        rows = []
        for t, (pair, pc) in P18_19.items():
            row = [t, f"{pair[0]} + {pair[1]} {fmt(pc, 3)}"]
            for name, md in res["models"].items():
                d = md[m]["define2"].get(t)
                if not d:
                    row.append("—")
                else:
                    row.append("<br>".join(f"{a} + {b} {fmt(c, 3)}" for a, b, c in d["top"]))
            rows.append(row)
        L.append(md_table(header, rows, align_right_from=99))
        rows = []
        for name, md in res["models"].items():
            nl = md[m]["null"]
            rows.append([name, fmt(nl["random_unit"]["mean"]), fmt(nl["random_unit"]["p90"]), fmt(nl["random_unit"]["max"]),
                         fmt(nl["random_word"]["mean"]), fmt(nl["random_word"]["p90"]), fmt(nl["random_word"]["max"])])
        L += ["", f"Null distribution of the best define2 score ({N_NULL} random unit-vector targets; {N_NULL} random real-word targets with the same filter):", "",
              md_table(["model", "unit mean", "unit p90", "unit max", "word mean", "word p90", "word max"], rows), ""]
    missing = [t for t in P18_19 if all(md[MODES[0]]["define2"].get(t) is None for md in res["models"].values())]
    if missing:
        L += [f"Targets marked — ({', '.join(missing)}) are not in the cached vocabulary: the notebooks fetched their vectors live from the retired OpenAI endpoint and never saved them, so the paper's numbers for them cannot be reproduced from the caches. Phase 3 embeds them with the open models.", ""]
    L += ["**Reading.** The paper's pp. 18–19 numbers came from two filter settings (no filter for puppy/duckling/foal; plural + prefix for the rest). "
          "With one filter, compare the Curie column to the paper column: pairs that change are ones the original filter let through. "
          "The random-word null is the honest comparison: a target's best pair only means something if its score is well above what an arbitrary vocabulary word gets. "
          "Random unit vectors score far lower in the raw spaces because real words occupy a narrow cone; after centering the two nulls converge.", ""]

    # analogies
    L += ["### 1.4 Analogies with the mikolov() bug fixed", "",
          "Score = cos(v(a) − v(b) + v(c), v(d)); rank excludes a, b, c. The notebooks' values were cos(·, v(\"gpt\")) and are not comparable.", ""]
    for m in MODES:
        rows = []
        names = list(res["models"].keys())
        for i, an in enumerate(res["models"][names[0]][m]["analogies"]):
            row = [an["analogy"]]
            for name in names:
                r = res["models"][name][m]["analogies"][i]
                row.append("skip" if "skipped" in r else f"{r['rank']} ({fmt(r['cos'], 3)}; top: {r['top'][0][0]})")
            rows.append(row)
        L += [f"**{m}** — cells are rank (cosine; top-1 word)", "", md_table(["analogy"] + names, rows, align_right_from=99), ""]
    L += [f"_Phase 1 ran in {res['seconds']:.0f} s._", ""]
    return "\n".join(L)
