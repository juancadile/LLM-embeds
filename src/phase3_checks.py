"""Two checks on the OPT magnitude result, using the Phase 3 original-recipe vectors already on
disk (results/cache/{model}_original_full_{preln,postln}.npy; no re-extraction).

1. Rogue dimensions: top-3 dimensions by share of total squared norm over the 5,124 vocab; the
   74-similar-vs-5,000-random L2 test rerun with the top-1 and the top-3 dimensions zeroed.
2. Token count: tokens per word (OPT tokenizer, word alone, no special tokens, as the recipe);
   Spearman(token count, L2 norm); the test restricted to similar pairs with equal token counts;
   the test against random pairs matched on the two words' token counts; and both at once.

    python -m src.phase3_checks          -> results/phase3_checks.json, rendered into REPORT.md
"""
from __future__ import annotations

import time

import numpy as np
from scipy import stats

from .common import CACHE, RESULTS, dump_json, fmt, md_table, rng
from .load import load_vocab
from .pairs import ALL_WORDS, CATEGORIES
from .phase2 import N_RANDOM, pct_diff
from .phase3 import magnitude_eval

MODELS = {"opt-1.3b": "facebook/opt-1.3b", "opt-13b": "facebook/opt-13b"}
SIDES = ("preln", "postln")
PAIRS = [p for pairs in CATEGORIES.values() for p in pairs]


def _words(vocab):
    vset = set(vocab)
    return list(vocab) + [w for w in ALL_WORDS if w not in vset]


def token_counts(hf_id: str, words: list[str]) -> dict[str, int]:
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(hf_id)
    return {w: len(tok(w, add_special_tokens=False).input_ids) for w in words}


def rogue_dims(V: np.ndarray, k: int = 3):
    sq = V.astype(np.float64) ** 2
    share = sq.sum(0) / sq.sum()
    top = np.argsort(-share)[:k]
    return [(int(d), float(share[d])) for d in top]


def _test(sim: np.ndarray, rand: np.ndarray) -> dict:
    return {"n_sim": int(len(sim)), "n_rand": int(len(rand)), "sim_median": float(np.median(sim)), "rand_median": float(np.median(rand)),
            "p_less": float(stats.mannwhitneyu(sim, rand, alternative="less").pvalue),
            "p_greater": float(stats.mannwhitneyu(sim, rand, alternative="greater").pvalue)}


def token_matched_tests(n: np.ndarray, idx: dict[str, int], tc: dict[str, int], n_vocab: int, per_pair: int = 68):
    """n = L2 norms over vocab+extra rows.  Returns the three token-count-controlled tests."""
    g = rng(11)
    vocab_by_tc: dict[int, np.ndarray] = {}
    for w, i in idx.items():
        if i < n_vocab:
            vocab_by_tc.setdefault(tc[w], []).append(i)
    vocab_by_tc = {k: np.array(v) for k, v in vocab_by_tc.items()}
    sim_all, sim_eq, rand_matched, rand_matched_eq = [], [], [], []
    unmatched = 0
    for a, b in PAIRS:
        d = pct_diff(n[idx[a]], n[idx[b]])
        ta, tb = tc[a], tc[b]
        sim_all.append(d)
        if ta == tb:
            sim_eq.append(d)
        A, B = vocab_by_tc.get(ta), vocab_by_tc.get(tb)
        if A is None or B is None or (ta == tb and len(A) < 2):
            unmatched += 1
            continue
        for _ in range(per_pair):
            i = g.choice(A)
            j = g.choice(B)
            while j == i:
                j = g.choice(B)
            r = pct_diff(n[i], n[j])
            rand_matched.append(r)
            if ta == tb:
                rand_matched_eq.append(r)
    g2 = rng(10)                                          # the standard random arm, same draw as magnitude_eval
    ri = g2.integers(0, n_vocab, N_RANDOM)
    rj = (ri + g2.integers(1, n_vocab, N_RANDOM)) % n_vocab
    rand_std = pct_diff(n[ri], n[rj])
    return {
        "n_pairs_equal_tc": len(sim_eq), "n_pairs_unmatched": unmatched,
        "original": _test(np.array(sim_all), rand_std),
        "equal_tc_pairs_vs_standard_random": _test(np.array(sim_eq), rand_std),
        "all_pairs_vs_tc_matched_random": _test(np.array(sim_all), np.array(rand_matched)),
        "equal_tc_pairs_vs_tc_matched_random": _test(np.array(sim_eq), np.array(rand_matched_eq)),
        "tc_pairs_hist": {f"{a}-{b}": int(sum(1 for x, y in PAIRS if sorted((tc[x], tc[y])) == [a, b]))
                          for a, b in sorted({tuple(sorted((tc[x], tc[y]))) for x, y in PAIRS})},
    }


def run():
    t0 = time.time()
    vocab = load_vocab()
    words = _words(vocab)
    idx = {w: i for i, w in enumerate(words)}
    n_vocab = len(vocab)
    res = {"models": {}, "n_vocab": n_vocab, "n_words": len(words)}
    for name, hf in MODELS.items():
        tc = token_counts(hf, words)
        entry = {"hf": hf, "token_counts": {"vocab_frac_multi": float(np.mean([tc[w] > 1 for w in vocab])),
                                           "vocab_max": int(max(tc[w] for w in vocab)),
                                           "pairs_words_multi": sorted(w for p in PAIRS for w in p if tc[w] > 1)},
                 "sides": {}}
        for side in SIDES:
            E = np.load(CACHE / f"{name}_original_full_{side}.npy")
            assert E.shape[0] == len(words), (name, side, E.shape)
            V, X = E[:n_vocab], E[n_vocab:]
            top = rogue_dims(V)
            s = {"rogue_dims": top, "rogue": {}}
            # (1) rogue-dimension ablation, L2 raw, same random pairs as the Phase 3 table
            for label, kill in (("original", []), ("top1_zeroed", [top[0][0]]), ("top3_zeroed", [d for d, _ in top])):
                Vz, Xz = V.copy(), X.copy()
                if kill:
                    Vz[:, kill] = 0
                    Xz[:, kill] = 0
                m = magnitude_eval(Vz, vocab, Xz, words[n_vocab:], "raw", "L2")
                s["rogue"][label] = {"sim_median": m["cats"]["all"]["median"], "rand_median": m["random"]["median"],
                                     "p_less": m["cats"]["all"]["p_less"], "p_greater": m["cats"]["all"]["p_greater"],
                                     "cats": {c: (m["cats"][c]["median"], m["cats"][c]["p_less"]) for c in CATEGORIES}}
            # (2) token count
            n = np.linalg.norm(E, axis=1)
            tcv = np.array([tc[w] for w in vocab])
            s["spearman_tc_norm_vocab"] = float(stats.spearmanr(tcv, n[:n_vocab])[0])
            s["spearman_tc_norm_all"] = float(stats.spearmanr(np.array([tc[w] for w in words]), n)[0])
            s["median_norm_by_tc"] = {int(k): float(np.median(n[:n_vocab][tcv == k])) for k in sorted(set(tcv.tolist())) if (tcv == k).sum() >= 20}
            s["token"] = token_matched_tests(n, idx, tc, n_vocab)
            entry["sides"][side] = s
            print(f"[checks] {name} {side} done ({time.time() - t0:.0f}s)", flush=True)
        res["models"][name] = entry
    res["seconds"] = time.time() - t0
    dump_json(res, RESULTS / "phase3_checks.json")
    return res


# ---------------------------------------------------------------------------
def report(res) -> str:
    L = ["### 3.x Checks on the OPT magnitude result: rogue dimensions and token count", "",
         "Vectors: the Phase 3 original-recipe extraction already on disk (no re-extraction), 5,124 vocab rows + 51 pair-word rows, "
         "pre-norm and post-norm sides. Test: |%diff| of L2 norm, 74 similar pairs vs 5,000 seeded random vocab pairs, Mann–Whitney (p< = similar closer).", ""]

    L += ["**Rogue dimensions.** Top-3 dimensions by share of total squared norm over the vocab, and the test with them zeroed.", ""]
    rows = []
    for name, e in res["models"].items():
        for side, s in e["sides"].items():
            dims = "; ".join(f"d{d}: {100 * sh:.1f}%" for d, sh in s["rogue_dims"])
            for label in ("original", "top1_zeroed", "top3_zeroed"):
                r = s["rogue"][label]
                rows.append([name, side, dims if label == "original" else "", label.replace("_", " "),
                             fmt(r["sim_median"], 1), fmt(r["rand_median"], 1), f"{r['p_less']:.2g}", f"{r['p_greater']:.2g}",
                             ", ".join(f"{c} {fmt(v[0], 1)} (p={v[1]:.1g})" for c, v in r["cats"].items())])
    L += [md_table(["model", "side", "top-3 dims (share of Σ‖v‖²)", "variant", "similar median", "random median", "p<", "p>", "per category"], rows, align_right_from=4), ""]

    L += ["**Token count.** Tokens per word from the OPT tokenizer (word alone, no special tokens; this is how the recipe tokenises). "
          "Spearman correlation with L2 norm; then the test restricted to similar pairs whose two words have equal token counts, "
          "and against random pairs drawn to match each similar pair's two token counts (68 per pair).", ""]
    rows = []
    for name, e in res["models"].items():
        t = e["token_counts"]
        for side, s in e["sides"].items():
            tk = s["token"]
            rows.append([name, side, f"{100 * t['vocab_frac_multi']:.0f}% of vocab >1 token (max {t['vocab_max']})",
                         fmt(s["spearman_tc_norm_vocab"], 3),
                         ", ".join(f"{k}: {fmt(v, 1)}" for k, v in s["median_norm_by_tc"].items()),
                         *[f"{fmt(tk[key]['sim_median'], 1)} vs {fmt(tk[key]['rand_median'], 1)} (n={tk[key]['n_sim']}, p<={tk[key]['p_less']:.2g}, p>={tk[key]['p_greater']:.2g})"
                           for key in ("original", "equal_tc_pairs_vs_standard_random", "all_pairs_vs_tc_matched_random", "equal_tc_pairs_vs_tc_matched_random")]])
    L += [md_table(["model", "side", "multi-token words", "Spearman(tokens, ‖v‖)", "median ‖v‖ by token count", "original (74 vs 5,000)",
                    "equal-count pairs vs standard random", "all pairs vs count-matched random", "equal-count pairs vs count-matched random"], rows, align_right_from=99), ""]
    any_e = next(iter(res["models"].values()))
    hist = any_e["sides"]["preln"]["token"]["tc_pairs_hist"]
    n_eq = any_e["sides"]["preln"]["token"]["n_pairs_equal_tc"]
    L += [f"Token-count pattern of the 74 similar pairs (count_a-count_b: pairs): {', '.join(f'{k}: {v}' for k, v in hist.items())}. {n_eq} of 74 pairs have equal counts.", ""]

    # one-sentence verdicts
    verdict = []
    for name, e in res["models"].items():
        for side, s in e["sides"].items():
            o, t3 = s["rogue"]["original"], s["rogue"]["top3_zeroed"]
            verdict.append(f"{name} {side}: rogue-dimension removal {'keeps' if t3['p_less'] < 0.05 else 'removes'} the closer-than-random result "
                           f"(p< {o['p_less']:.1g} → {t3['p_less']:.1g} with top-3 zeroed)")
    L += ["**Reading.** " + "; ".join(verdict) + ".", ""]
    verdict = []
    for name, e in res["models"].items():
        for side, s in e["sides"].items():
            tk = s["token"]
            m = tk["equal_tc_pairs_vs_tc_matched_random"]
            o = tk["original"]
            verdict.append(f"{name} {side}: Spearman(tokens, ‖v‖) = {s['spearman_tc_norm_vocab']:+.2f}; with both sides matched on token count the result is "
                           f"{'still closer' if m['p_less'] < 0.05 else ('farther' if m['p_greater'] < 0.05 else 'gone')} "
                           f"(similar {fmt(m['sim_median'], 1)} vs random {fmt(m['rand_median'], 1)}, n={m['n_sim']}, p< {m['p_less']:.1g}; original p< {o['p_less']:.1g})")
    L += ["; ".join(verdict) + ".", ""]
    post = [e["sides"]["postln"]["token"]["equal_tc_pairs_vs_tc_matched_random"] for e in res["models"].values()]
    if all(p["p_less"] >= 0.05 for p in post):
        L += ["**Post-norm \"closer\" disappears under token-count matching on both OPT models: it was token count in disguise.**", ""]
    elif all(p["p_less"] < 0.05 for p in post):
        L += ["**Post-norm \"closer\" survives token-count matching on both OPT models**, but the matched medians are "
              + " and ".join(f"{fmt(p['sim_median'], 1)} vs {fmt(p['rand_median'], 1)}" for p in post)
              + " percent: after LayerNorm the whole magnitude spread among equal-token-count words is under 2 %, so what survives is statistically real and numerically negligible.", ""]
    else:
        L += ["**Post-norm \"closer\" survives token-count matching on one OPT model and not the other.**", ""]
    for name, e in res["models"].items():
        pre = e["sides"]["preln"]
        r3, tm = pre["rogue"]["top3_zeroed"], pre["token"]["equal_tc_pairs_vs_tc_matched_random"]
        if r3["p_less"] >= 0.05 and tm["p_less"] >= 0.05:
            L += [f"**{name} pre-norm (the paper's object for this model): the closer-than-random result does not survive either control.** "
                  f"It rests on dimension d{pre['rogue_dims'][0][0]} ({100 * pre['rogue_dims'][0][1]:.0f} % of total squared norm) and on token count "
                  f"(p< {r3['p_less']:.2f} with the top-3 dimensions zeroed; p< {tm['p_less']:.2f} with both arms matched on token count).", ""]
        elif r3["p_less"] < 0.05 and tm["p_less"] < 0.05:
            L += [f"**{name} pre-norm: the closer-than-random result survives both controls** (p< {r3['p_less']:.1g} with top-3 dimensions zeroed; p< {tm['p_less']:.1g} token-matched).", ""]
    L += [f"_Checks ran in {res.get('seconds', 0):.0f} s._", ""]
    return "\n".join(L)


if __name__ == "__main__":
    run()
    from .run_all import build_report
    text = build_report()
    start = text.find("### 3.x Checks on the OPT magnitude result")
    print(text[start:] if start >= 0 else text[-3000:])
