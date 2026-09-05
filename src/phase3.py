"""Phase 3: a 2026 open model, with every extraction choice made explicit, plus the GPU
re-extraction that fills the magnitude-pair gaps for the original OPT/T5 models.

3.1  extract vocab (+ magnitude-pair words) from each HF model under
     layer × pooling × context; 3.2 rerun the p.16 table, controls, define2 and the
     magnitude test on every configuration; 3.3 score configurations.
"""
from __future__ import annotations

import json
import os
import shutil
import time
from pathlib import Path

import numpy as np
from scipy import stats

from .common import CACHE, RESULTS, dump_json, fmt, md_table, rng
from .extract import CONTEXTS, LAYERS, POOLS, cfg_name, extract, extract_original_recipe, slug
from .filters import cognate_indices
from .load import MAGNITUDE_MODELS, load, load_vocab
from .pairs import ALL_WORDS, CATEGORIES
from .phase1 import BACHELOR, P16, P18_19, controls, random_pair_baseline
from .phase2 import N_RANDOM, pct_diff, norms
from .similarity import Space, center

DEFAULT_MODELS = ["Qwen/Qwen3-14B", "meta-llama/Llama-3.1-8B-Instruct"]
ORIGINAL = {  # notebook models, for the magnitude re-extraction; approx. download size in GB
    "opt-1.3b": ("facebook/opt-1.3b", 3), "opt-13b": ("facebook/opt-13b", 26),
    "t5-large": ("t5-large", 3), "t5-3b": ("t5-3b", 11), "flan-t5-xxl": ("google/flan-t5-xxl", 45),
}
DOWNLOAD_LIMIT_GB = 40
MIN_FREE_GB = 20


def _free_gb(path=".") -> float:
    return shutil.disk_usage(path).free / 1e9


def _hf_cached(model_id: str) -> bool:
    hub = Path(os.environ.get("HF_HOME", Path.home() / ".cache/huggingface")) / "hub"
    return (hub / f"models--{model_id.replace('/', '--')}").exists()


# ---------------------------------------------------------------------------
def magnitude_eval(vocab_V: np.ndarray, vocab: list[str], extra_V: np.ndarray, extra_words: list[str],
                   mode: str, metric: str = "L2"):
    """Similar pairs (from vocab + extra rows) vs random vocab pairs, one metric/mode."""
    X = np.vstack([vocab_V, extra_V])
    words = list(vocab) + list(extra_words)
    if mode == "center":
        X = X - vocab_V.mean(0, keepdims=True)         # centre with the vocab mean only
    idx = {w: i for i, w in enumerate(words)}
    n = norms(X, metric)
    g = rng(10)
    N = len(vocab)
    ri = g.integers(0, N, N_RANDOM)
    rj = (ri + g.integers(1, N, N_RANDOM)) % N
    rd = pct_diff(n[ri], n[rj])
    out = {"random": {"mean": float(rd.mean()), "median": float(np.median(rd))}, "cats": {}}
    pooled = []
    for cat, pairs in CATEGORIES.items():
        d = np.array([pct_diff(n[idx[a]], n[idx[b]]) for a, b in pairs if a in idx and b in idx])
        pooled.extend(d.tolist())
        out["cats"][cat] = {"n": len(d), "mean": float(d.mean()), "median": float(np.median(d)),
                            "p_less": float(stats.mannwhitneyu(d, rd, alternative="less").pvalue)}
    d = np.array(pooled)
    out["cats"]["all"] = {"n": len(d), "mean": float(d.mean()), "median": float(np.median(d)),
                          "p_less": float(stats.mannwhitneyu(d, rd, alternative="less").pvalue),
                          "p_greater": float(stats.mannwhitneyu(d, rd, alternative="greater").pvalue)}
    return out


def eval_config(vocab, V, extra_V, extra_words, name):
    """Phase-1 steps 3–5 and Phase-2 step 2 on one configuration, raw and centred."""
    res = {}
    extra_idx = {w: i for i, w in enumerate(extra_words)}
    for mode in ("raw", "center"):
        S = Space.from_raw(vocab, V, name, mode)
        mu = V.mean(0) if mode == "center" else 0.0
        p16 = []
        for a, b, t in P16 + [BACHELOR]:
            top = S.define_ab(a, b, 5, exclude_ab=True)
            rank, cos = S.rank_of(S.vec(a) + S.vec(b), t, exclude=(a, b))
            p16.append({"a": a, "b": b, "target": t, "rank": rank, "cos": cos, "top": top[:3]})
        d2 = {}
        for t, (pair, _) in P18_19.items():
            ex = cognate_indices(t, vocab)
            if t in S:
                top = S.define2_word(t, 3, ex)
            elif t in extra_idx:          # target not in the vocab (e.g. 'foal'): query with its extracted vector
                top = S.define2(extra_V[extra_idx[t]] - mu, 3, ex)
            else:
                d2[t] = None
                continue
            d2[t] = {"top": top, "paper_pair_hit": set(top[0][:2]) == set(pair)}
        res[mode] = {
            "p16": p16, "controls": controls(S, P16 + [BACHELOR]),
            "baseline": random_pair_baseline(S, n=300),
            "define2": d2,
            "magnitude": {m: magnitude_eval(V, vocab, extra_V, extra_words, mode, m) for m in ("L1", "L2")},
        }
    return res


def score(cfg_res) -> dict:
    """Headline numbers used to rank configurations (raw mode)."""
    r = cfg_res["raw"]
    seven = [x for x in r["p16"] if x["target"] != "bachelor"]
    ctrl = {c["target"]: c for c in r["controls"]}
    return {
        "p16_rank1": sum(x["rank"] == 1 for x in seven),
        "p16_mean_rank": float(np.mean([x["rank"] for x in seven])),
        "sum_beats_b": sum(ctrl[x["target"]]["rank_a+b"] < ctrl[x["target"]]["rank_b"] for x in seven),
        "bachelor_rank": ctrl["bachelor"]["rank_a+b"],
        "define2_paper_hits": sum(v["paper_pair_hit"] for v in r["define2"].values()),
        "mag_L2_p_less": r["magnitude"]["L2"]["cats"]["all"]["p_less"],
        "mag_L2_median_ratio": r["magnitude"]["L2"]["cats"]["all"]["median"] / max(r["magnitude"]["L2"]["random"]["median"], 1e-9),
        "baseline_mean": r["baseline"]["mean"],
    }


# ---------------------------------------------------------------------------
def run(models=None, skip_original=False):
    import torch
    t0 = time.time()
    res = {"models": {}, "original": {}, "notes": []}
    vocab = load_vocab()
    vset = set(vocab)
    extra_words = sorted({w for w in ALL_WORDS if w not in vset} | {t for t in P18_19 if t not in vset})
    all_words = vocab + extra_words
    models = models or [m for m in DEFAULT_MODELS if _hf_cached(m)] or DEFAULT_MODELS[:1]
    if not torch.cuda.is_available():
        res["notes"].append("CUDA not available; Phase 3 skipped.")
        dump_json(res, RESULTS / "phase3.json")
        return res

    # reuse fully evaluated models from a previous (interrupted) run
    prev_path = RESULTS / "phase3.json"
    prev = json.load(open(prev_path)) if prev_path.exists() else {}
    n_cfg = len(LAYERS) * len(POOLS) * len(CONTEXTS)

    # 3.1 + 3.2: 2026 models
    for mid in models:
        if mid in prev.get("models", {}) and len(prev["models"][mid].get("configs", {})) == n_cfg:
            res["models"][mid] = prev["models"][mid]
            res["notes"].append(f"{mid}: {n_cfg} configurations reused from the previous run's phase3.json.")
            print(f"[phase3] {mid}: reused", flush=True)
            continue
        if _free_gb() < MIN_FREE_GB:
            res["notes"].append(f"Stopped before {mid}: free disk {_free_gb():.0f} GB < {MIN_FREE_GB}.")
            break
        try:
            ex = extract(mid, all_words, batch_size=32, n_vocab=len(vocab))
        except Exception as e:  # noqa: BLE001
            res["notes"].append(f"{mid}: extraction failed: {type(e).__name__}: {str(e)[:300]}")
            print(res["notes"][-1], flush=True)
            continue
        entry = {"tokens": ex["tokens"], "extract_seconds": ex.get("seconds"), "layer_indices": ex.get("layer_indices"), "configs": {}}
        for layer in LAYERS:
            for pool in POOLS:
                for ctx in CONTEXTS:
                    name = cfg_name(mid, layer, pool, ctx)
                    A = np.load(CACHE / f"{name}.npy")
                    V, extra_V = A[:len(vocab)], A[len(vocab):]
                    cr = eval_config(vocab, V, extra_V, extra_words, name)
                    cr["score"] = score(cr)
                    entry["configs"][name] = cr
                    print(f"[phase3] {name} p16_rank1={cr['score']['p16_rank1']} ({time.time() - t0:.0f}s)", flush=True)
        res["models"][mid] = entry
        dump_json(res, RESULTS / "phase3.json")

    # 3.4: original OPT/T5 recipe for the magnitude pairs (fills Phase-2 coverage gaps)
    if not skip_original:
        for name, (mid, gb) in ORIGINAL.items():
            if _free_gb() < MIN_FREE_GB + gb:
                res["notes"].append(f"{name}: skipped, would leave < {MIN_FREE_GB} GB free.")
                continue
            if gb > DOWNLOAD_LIMIT_GB and not _hf_cached(mid):
                res["notes"].append(f"{name}: skipped, download ≈{gb} GB exceeds the {DOWNLOAD_LIMIT_GB} GB limit set in the brief.")
                continue
            try:
                # One extraction run covers the whole vocab AND the pair words, so the similar-pair
                # arm and the random arm of the magnitude test come from the same vectors.
                cvocab, CV = load(name)
                orig_words = list(cvocab) + [w for w in ALL_WORDS if w not in set(cvocab)]
                sides = {}
                for side in ("preln", "postln"):
                    p = CACHE / f"{name}_original_full_{side}.npy"
                    if p.exists() and np.load(p, mmap_mode="r").shape[0] == len(orig_words):
                        sides[side] = np.load(p)
                if len(sides) < 2:
                    sides = extract_original_recipe(mid, orig_words, f"{name}_original_full")
                extra = orig_words[len(cvocab):]
                entry = {"hf": mid, "n_vocab": len(cvocab), "n_extra": len(extra), "sides": {}}
                for side, E in sides.items():
                    EV, EX = E[:len(cvocab)], E[len(cvocab):]
                    # consistency of this side with the cache, over all 5,124 vocab words
                    cos = np.einsum("ij,ij->i", CV, EV) / (np.linalg.norm(CV, axis=1) * np.linalg.norm(EV, axis=1) + 1e-9)
                    ratio = np.linalg.norm(EV, axis=1) / (np.linalg.norm(CV, axis=1) + 1e-9)
                    nE = np.linalg.norm(EV, axis=1)
                    mag = {m: {mode: magnitude_eval(EV, cvocab, EX, extra, mode, m) for mode in ("raw", "center")} for m in ("L1", "L2")}
                    entry["sides"][side] = {
                        "consistency": {"cos_mean": float(cos.mean()), "cos_median": float(np.median(cos)), "cos_p5": float(np.percentile(cos, 5)),
                                        "cos_min": float(cos.min()), "frac_cos_below_0.99": float((cos < 0.99).mean()),
                                        "norm_ratio_mean": float(ratio.mean()), "norm_ratio_p5": float(np.percentile(ratio, 5)),
                                        "norm_ratio_p95": float(np.percentile(ratio, 95)), "norm_ratio_cv": float(ratio.std() / ratio.mean()),
                                        "norm_cv_this_side": float(nE.std() / nE.mean()),
                                        "norm_cv_cache": float(np.linalg.norm(CV, axis=1).std() / np.linalg.norm(CV, axis=1).mean())},
                        "magnitude": mag}
                    print(f"[phase3] original {name} {side}: cache cos median {np.median(cos):.4f}, norm ratio {ratio.mean():.3f} (cv {ratio.std() / ratio.mean():.3f}) ({time.time() - t0:.0f}s)", flush=True)
                res["original"][name] = entry
            except Exception as e:  # noqa: BLE001
                res["original"][name] = {"error": f"{type(e).__name__}: {str(e)[:300]}"}
                print(f"[phase3] original {name} failed: {e}", flush=True)
            dump_json(res, RESULTS / "phase3.json")
    res["seconds"] = time.time() - t0
    dump_json(res, RESULTS / "phase3.json")
    return res


# ---------------------------------------------------------------------------
def report(res) -> str:
    L = ["## Phase 3 — 2026 open model with explicit extraction choices", ""]
    for n in res.get("notes", []):
        L.append(f"> {n}")
    if res.get("notes"):
        L.append("")
    for mid, entry in res.get("models", {}).items():
        L += [f"### 3.{list(res['models']).index(mid) + 1} {mid}", "",
              f"Layer indices used: {entry.get('layer_indices')}. Extraction {(entry.get('extract_seconds') or 0):.0f} s (0 = reused from cache). "
              + "; ".join(
                  f"{ctx}: {sum(v > 1 for v in tk.values())}/{len(tk)} words need >1 token, worst "
                  + ", ".join(f"{k}={v}" for k, v in sorted(tk.items(), key=lambda x: -x[1])[:5])
                  for ctx, tk in entry["tokens"].items() if isinstance(tk, dict)), ""]
        rows = []
        for name, cr in entry["configs"].items():
            s = cr["score"]
            _, layer, pool, ctx = name.rsplit("_", 3)
            c7 = [x for x in cr["center"]["p16"] if x["target"] != "bachelor"]
            c_rank1 = sum(x["rank"] == 1 for x in c7)
            c_mean = float(np.mean([x["rank"] for x in c7]))
            c_hits = sum(bool(v and v["paper_pair_hit"]) for v in cr["center"]["define2"].values())
            rows.append([layer, pool, ctx, f"{s['p16_rank1']} / {c_rank1}", f"{fmt(s['p16_mean_rank'], 1)} / {fmt(c_mean, 1)}",
                         s["sum_beats_b"], s["bachelor_rank"], f"{s['define2_paper_hits']} / {c_hits}",
                         f"{s['mag_L2_p_less']:.2g}", fmt(s["mag_L2_median_ratio"], 2), fmt(s["baseline_mean"], 3),
                         s["p16_rank1"], c_rank1, s["p16_mean_rank"]])
        rows.sort(key=lambda r: (-max(r[-3], r[-2]), -r[-3], r[-1]))
        rows = [r[:-3] for r in rows]
        L += [md_table(["layer", "pool", "context", "p16 targets at rank 1 (of 7), raw / centered", "mean rank, raw / centered", "sum beats v(b) alone (of 7)", "bachelor rank",
                        "define2 = paper pair (of 10), raw / centered", "magnitude p (similar < random, L2 raw)", "median |%diff| ratio similar/random", "random-sum baseline (raw)"], rows, align_right_from=3), ""]
        best = rows[0]
        bname = cfg_name(mid, best[0], best[1], best[2])
        cr = entry["configs"][bname]["raw"]
        L += [f"**Best configuration by the p. 16 criterion: layer {best[0]}, {best[1]} pooling, context {best[2]}.** Its p. 16 table (raw, a and b excluded):", ""]
        L.append(md_table(["a + b → target", "rank", "cos", "top-3"],
                          [[f"{x['a']} + {x['b']} → {x['target']}", x["rank"], fmt(x["cos"]), ", ".join(f"{w} {fmt(c, 3)}" for w, c in x["top"])] for x in cr["p16"]]))
        L += ["", "define2 (raw) on the same configuration:", "",
              md_table(["target", "paper (Curie)", "this model top-3"],
                       [[t, f"{p[0]} + {p[1]}", "<br>".join(f"{a} + {b} {fmt(c, 3)}" for a, b, c in cr["define2"][t]["top"])] for t, (p, _) in P18_19.items()], align_right_from=99), ""]
        # magnitude across configs, L2 raw
        rows = []
        for name, c in entry["configs"].items():
            _, layer, pool, ctx = name.rsplit("_", 3)
            for mode in ("raw", "center"):
                m = c[mode]["magnitude"]["L2"]
                rows.append([layer, pool, ctx, mode, f"{fmt(m['random']['median'], 1)}",
                             *[f"{fmt(m['cats'][cat]['median'], 1)} (p={m['cats'][cat]['p_less']:.2g})" for cat in list(CATEGORIES) + ['all']]])
        L += ["Magnitude test on every configuration (L2, median |%diff|; p = similar < random):", "",
              md_table(["layer", "pool", "ctx", "mode", "random"] + list(CATEGORIES) + ["all"], rows, align_right_from=99), ""]

    if res.get("original"):
        L += ["### 3.x Original OPT/T5 recipe re-extracted on the GPU: the clean 74-pair magnitude test, on both sides of the final layer norm", "",
              "Word alone, final layer, mean over tokens (OPT drops BOS; T5 keeps EOS), as in the notebooks. The whole 5,124-word vocab and the 51 "
              "out-of-vocab pair words are embedded in one run, so the similar-pair arm and the 5,000 random pairs come from the same vectors. "
              "A forward hook captures the model's final normalisation layer on both sides: **preln** is the residual stream before it, **postln** is after it "
              "(what current transformers returns as `last_hidden_state`). Consistency = cosine and norm ratio between that side and the cached vector over all 5,124 vocab words: "
              "the side the cache matches tells which object the paper's magnitude numbers were computed on. "
              "Post-LN norms carry little per-word information by construction (each token is normalised to unit variance, then scaled by γ), so a null magnitude result on that side is trivial; the pre-LN side is where magnitude could mean anything.", ""]
        rows = []
        for name, o in res["original"].items():
            if "error" in o:
                rows.append([name, "", o["error"], "", "", "", "", ""])
                continue
            for side, s in o["sides"].items():
                c = s["consistency"]
                m = s["magnitude"]["L2"]["raw"]
                rows.append([name, side,
                             f"cos median {fmt(c['cos_median'], 4)} (p5 {fmt(c['cos_p5'], 3)}, min {fmt(c['cos_min'], 3)}); norm ratio {fmt(c['norm_ratio_mean'], 3)} (p5 {fmt(c['norm_ratio_p5'], 3)}, p95 {fmt(c['norm_ratio_p95'], 3)}, cv {fmt(c['norm_ratio_cv'], 3)}); norm cv this side {fmt(c['norm_cv_this_side'], 3)} vs cache {fmt(c['norm_cv_cache'], 3)}",
                             fmt(m["random"]["median"], 1),
                             *[f"{fmt(m['cats'][c_]['median'], 1)} (n={m['cats'][c_]['n']}, p={m['cats'][c_]['p_less']:.2g})" for c_ in list(CATEGORIES) + ["all"]]])
        L += [md_table(["model", "side", "this side vs cache", "random median (L2 raw)"] + list(CATEGORIES) + ["all pooled"], rows, align_right_from=99), ""]
        rows = []
        for name, o in res["original"].items():
            if "error" in o:
                continue
            for side, s in o["sides"].items():
                for metric in ("L1", "L2"):
                    for mode in ("raw", "center"):
                        m = s["magnitude"][metric][mode]["cats"]["all"]
                        rows.append([name, side, metric, mode, fmt(m["median"], 1), fmt(s["magnitude"][metric][mode]["random"]["median"], 1), f"{m['p_less']:.2g}", f"{m['p_greater']:.2g}"])
        L += ["All metrics, pooled over the 74 pairs:", "", md_table(["model", "side", "metric", "mode", "similar median", "random median", "p similar<random", "p similar>random"], rows, align_right_from=4), ""]

    checks = RESULTS / "phase3_checks.json"
    if checks.exists():
        from . import phase3_checks
        L += [phase3_checks.report(json.load(open(checks)))]

    L += ["### Recommendation for the method section", "",
          "_See the Reading paragraphs above; the recommendation paragraph is written by hand in results/NOTES.md once the tables are in._", "",
          f"_Phase 3 ran in {res.get('seconds', 0):.0f} s._", ""]
    return "\n".join(L)
