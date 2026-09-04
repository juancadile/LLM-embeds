"""Phase 2: the magnitude study, redone.

For each cached model: symmetric percent difference in magnitude,
|n1 − n2| / mean(n1, n2) × 100, for the similar-word pairs vs. seeded random pairs,
in L1 and L2, raw and mean-centred, with Mann–Whitney U tests (alternative: similar
pairs have *smaller* differences).  Also token counts per word, the L1–L2 correlation,
and the share of squared norm held by the top-variance dimensions.

Only pairs whose two words are both in the 5,124-word cache vocabulary can be scored
from the caches; the notebooks embedded the rest on the fly and did not save them.
Coverage is reported per category.  ``phase2_extra`` (Phase 3 module) can fill the
gaps by re-extracting the missing words on the GPU.
"""
from __future__ import annotations

import time

import numpy as np
from scipy import stats

from .common import CACHE, RESULTS, dump_json, fmt, md_table, rng
from .load import MAGNITUDE_MODELS, load
from .pairs import ALL_WORDS, CATEGORIES
from .similarity import center

N_RANDOM = 5000
METRICS = ("L1", "L2")
MODES = ("raw", "center")

TOKENIZERS = {
    "opt-1.3b": "facebook/opt-1.3b", "opt-13b": "facebook/opt-13b",
    "t5-large": "t5-large", "t5-3b": "t5-3b", "flan-t5-xxl": "google/flan-t5-xxl",
}


def norms(V: np.ndarray, metric: str) -> np.ndarray:
    return np.abs(V).sum(1) if metric == "L1" else np.linalg.norm(V, axis=1)


def pct_diff(n1: np.ndarray, n2: np.ndarray) -> np.ndarray:
    return np.abs(n1 - n2) / ((n1 + n2) / 2) * 100


def token_counts(model: str, words: list[str]) -> dict[str, int]:
    try:
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained(TOKENIZERS[model])
        return {w: len(tok(w, add_special_tokens=False).input_ids) for w in words}
    except Exception as e:  # offline or missing tokenizer
        return {"_error": str(e)[:200]}


def spectrum(V: np.ndarray, k: int = 5):
    """Top-k dimensions by variance and the mean share of squared L2 norm they hold."""
    var = V.var(axis=0)
    top = np.argsort(-var)[:k]
    sq = V.astype(np.float64) ** 2
    share = sq / sq.sum(1, keepdims=True)
    return [{"dim": int(d), "var_share": float(var[d] / var.sum()), "sqnorm_share": float(share[:, d].mean()),
             "sqnorm_share_median": float(np.median(share[:, d])), "pooled_share": float(sq[:, d].sum() / sq.sum())} for d in top]


def run_model(name: str):
    vocab, V = load(name)
    idx = {w: i for i, w in enumerate(vocab)}
    g = rng(10)
    N = len(vocab)
    rand_i = g.integers(0, N, N_RANDOM)
    rand_j = (rand_i + g.integers(1, N, N_RANDOM)) % N     # never the same word

    res = {"n_vocab": N, "coverage": {}, "tests": {}, "pairs": {}, "random": {}}
    for cat, pairs in CATEGORIES.items():
        ok = [(a, b) for a, b in pairs if a in idx and b in idx]
        res["coverage"][cat] = {"n_pairs": len(pairs), "n_in_vocab": len(ok), "pairs_in_vocab": ok,
                                "missing_words": sorted({w for p in pairs for w in p if w not in idx})}

    for mode in MODES:
        X = V if mode == "raw" else center(V)
        for metric in METRICS:
            n = norms(X, metric)
            rd = pct_diff(n[rand_i], n[rand_j])
            key = f"{metric}_{mode}"
            res["random"][key] = {"mean": float(rd.mean()), "median": float(np.median(rd)), "n": N_RANDOM}
            pooled = []
            res["tests"][key] = {}
            res["pairs"][key] = {}
            for cat in CATEGORIES:
                ok = res["coverage"][cat]["pairs_in_vocab"]
                if not ok:
                    res["tests"][key][cat] = None
                    continue
                d = np.array([pct_diff(n[idx[a]], n[idx[b]]) for a, b in ok])
                pooled.extend(d.tolist())
                res["pairs"][key][cat] = [(a, b, float(x)) for (a, b), x in zip(ok, d)]
                p = stats.mannwhitneyu(d, rd, alternative="less").pvalue
                res["tests"][key][cat] = {"n": len(d), "mean": float(d.mean()), "median": float(np.median(d)), "p_less": float(p)}
            d = np.array(pooled)
            res["tests"][key]["all"] = {"n": len(d), "mean": float(d.mean()), "median": float(np.median(d)),
                                        "p_less": float(stats.mannwhitneyu(d, rd, alternative="less").pvalue),
                                        "p_greater": float(stats.mannwhitneyu(d, rd, alternative="greater").pvalue)}
    l1, l2 = norms(V, "L1"), norms(V, "L2")
    res["l1_l2"] = {"pearson": float(stats.pearsonr(l1, l2)[0]), "spearman": float(stats.spearmanr(l1, l2)[0])}
    res["spectrum"] = spectrum(V)
    res["tokens"] = token_counts(name, ALL_WORDS)
    return res


def run(models=MAGNITUDE_MODELS):
    t0 = time.time()
    res = {"models": {}}
    for name in models:
        res["models"][name] = run_model(name)
        print(f"[phase2] {name} done ({time.time() - t0:.0f}s)", flush=True)
    from .notebook_record import analyse
    res["notebook_record"] = analyse(include_typos=False)
    res["seconds"] = time.time() - t0
    dump_json(res, RESULTS / "phase2.json")
    return res


# ---------------------------------------------------------------------------
def _stars(p):
    return "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else ""


def report(res) -> str:
    L = ["## Phase 2 — magnitude study, redone on the cached embeddings", ""]
    models = list(res["models"].keys())

    # coverage
    rows = []
    for name in models:
        cov = res["models"][name]["coverage"]
        rows.append([name] + [f"{cov[c]['n_in_vocab']}/{cov[c]['n_pairs']}" for c in CATEGORIES])
    L += ["### 2.1 Coverage: pairs with both words in the cached vocabulary", "",
          md_table(["model"] + list(CATEGORIES), rows), "",
          "The notebooks embedded out-of-vocabulary words on the fly and never saved them, so from the caches only these pairs can be scored. "
          "Missing words are listed in results/phase2.json; the GPU re-extraction in Phase 3 (`phase2_extra`) fills them where the original model could be loaded.", ""]

    # main tables
    for key in [f"{m}_{mode}" for mode in MODES for m in METRICS]:
        metric, mode = key.split("_")
        rows = []
        for name in models:
            r = res["models"][name]
            rd = r["random"][key]
            row = [name, f"{fmt(rd['mean'], 1)} / {fmt(rd['median'], 1)}"]
            for cat in list(CATEGORIES) + ["all"]:
                t = r["tests"][key].get(cat)
                row.append("—" if not t else f"{fmt(t['mean'], 1)} / {fmt(t['median'], 1)} (n={t['n']}, p={t['p_less']:.3g}{_stars(t['p_less'])})")
            rows.append(row)
        L += [f"### 2.2 |%diff| in {metric}, {mode} — similar pairs vs {N_RANDOM} random pairs (mean / median; p = Mann–Whitney, similar < random)", "",
              md_table(["model", "random"] + list(CATEGORIES) + ["all pooled"], rows, align_right_from=99), ""]

    # verdict paragraph, from L2 raw pooled
    verdict = []
    for name in models:
        t = res["models"][name]["tests"]["L2_raw"]["all"]
        rd = res["models"][name]["random"]["L2_raw"]
        if t["p_less"] < 0.05:
            verdict.append(f"{name}: similar pairs closer (median {fmt(t['median'], 1)} vs {fmt(rd['median'], 1)}, p={t['p_less']:.2g})")
        elif t["p_greater"] < 0.05:
            verdict.append(f"{name}: similar pairs *farther* (median {fmt(t['median'], 1)} vs {fmt(rd['median'], 1)}, p={t['p_greater']:.2g})")
        else:
            verdict.append(f"{name}: no significant difference (median {fmt(t['median'], 1)} vs {fmt(rd['median'], 1)})")
    L += ["**Reading (L2, raw, pooled).** " + "; ".join(verdict) + ". ",
          "The paper's sentence \"similar pairs were not closer in magnitude than random pairs\" is a claim about the pooled comparison; "
          "where p < 0.05 in the \"closer\" direction it is contradicted on the authors' own pairs. "
          "Where results differ across models, the defensible statement is \"no consistent effect across models\". "
          "None of this tests a complexity–magnitude *correlation*: the design only checks that magnitude is invariant under near-synonymy, a necessary condition.", ""]

    # the notebooks' own printed values, all 74 pairs: record only, no test
    if res.get("notebook_record"):
        rows = []
        for name, e in res["notebook_record"].items():
            row = [name, f"{e['n_used']}/{e['n_pairs_parsed']}"]
            for cat in list(CATEGORIES) + ["all"]:
                t = e["cats"].get(cat)
                row.append("—" if not t else f"{fmt(t['mean'], 1)} / {fmt(t['median'], 1)} (n={t['n']})")
            row.append(", ".join(f"{x['pair']} {x['pct']:+.0f}%" for x in e["largest"]))
            rows.append(row)
        L += ["### 2.2b The notebooks' own printed values for the 74 similar pairs — record only, NOT a valid test", "",
              "Parsed from the saved cell outputs of magnitudes*.ipynb: signed L1 percent difference |n1/n2 − 1|·100 per similar pair, computed by the notebooks from vectors they embedded on the fly (only 10 of the 74 pairs are in the cache). Typo pairs excluded.", "",
              "**Invalid as a test, and the reviewer's letter must not cite it.** No random arm exists in the notebook outputs (they print only a signed 100-word mean per word, which cancels). Any random baseline for these values has to come from the cached matrices, so a test would compare on-the-fly vectors against cached vectors. `opt/1_3B.txt.orig` shows the cache was re-extracted at least once, so the two sets are not guaranteed to be the same extraction, and a p-value from mixed arms is not evidence. The same mixing was inside the notebooks' own `Rand100` column. The only clean 74-pair comparison is the GPU re-extraction in Phase 3, which embeds the similar pairs *and* the random pairs with one recipe in one run and reports the cosine between re-extracted and cached vectors for the words present in both.", "",
              md_table(["model", "pairs parsed", *[f"{c} mean / median |%diff|" for c in CATEGORIES], "all", "largest |diff|"], rows, align_right_from=99), ""]

    # tokens
    L += ["### 2.3 Tokens per word (tokenizer of each model family, word alone, no special tokens)", ""]
    rows = []
    for name in models:
        tk = res["models"][name]["tokens"]
        if "_error" in tk:
            rows.append([name, "tokenizer unavailable: " + tk["_error"]])
            continue
        multi = sorted(((w, n) for w, n in tk.items() if n > 1), key=lambda x: -x[1])
        rows.append([name, ", ".join(f"{w}={n}" for w, n in multi[:14]) + (" …" if len(multi) > 14 else "")])
    L += [md_table(["model", "words with >1 token (count)"], rows, align_right_from=99), "",
          "Because the cached OPT/T5 vectors are means over subword tokens (T5 also averages in the EOS token), a pair whose members differ in token count is not a clean magnitude comparison. Outliers in the pair tables above line up with these words.", ""]

    # L1 vs L2, spectrum
    rows = []
    for name in models:
        r = res["models"][name]
        sp = r["spectrum"]
        rows.append([name, fmt(r["l1_l2"]["pearson"], 3), fmt(r["l1_l2"]["spearman"], 3),
                     "; ".join(f"d{s['dim']}: mean {100 * s['sqnorm_share']:.1f}%, median {100 * s['sqnorm_share_median']:.1f}%" for s in sp[:3])])
    L += ["### 2.4 L1 vs L2 and rogue dimensions", "",
          md_table(["model", "Pearson(L1, L2)", "Spearman", "top-3 variance dims: share of a word's squared L2 norm (mean, median over words)"], rows, align_right_from=99), "",
          "The notebooks' \"magnitude\" was L1 (sum of absolute components). Where L1 and L2 are weakly correlated, the choice of norm changes the result; "
          "a dimension holding a large share of squared norm on its own is a rogue dimension (Timkey & van Schijndel 2021) and dominates L2 but not L1.", "",
          f"_Phase 2 ran in {res['seconds']:.0f} s._", ""]
    return "\n".join(L)
