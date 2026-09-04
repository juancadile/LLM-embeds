"""Reproduce the notebooks' OPT extraction under transformers 4.20.1 (the pinned version in
environment.yml) and compare with the cache and with the current-transformers re-extraction.

Run inside a throwaway venv that has transformers==4.20.1:
    python -m src.repro_tf420 [n_words]
Uses the notebook function verbatim (old/opt_squeeze.py: OPTModel, word alone, last_hidden_state,
drop BOS, mean over tokens) on a pytorch_model.bin snapshot, in float32 on CPU.
Writes results/repro_tf420.json.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent


def main(n: int = 20):
    import torch
    import transformers
    from transformers import GPT2Tokenizer, OPTModel

    vocab = [w.strip() for w in open(ROOT / "vocab/expanded_vocab.txt") if w.strip()]
    rng = np.random.default_rng(2026)
    idx = sorted(rng.choice(len(vocab), n, replace=False).tolist())
    words = [vocab[i] for i in idx]

    # cache rows for those words
    import pandas as pd
    C = pd.read_csv(ROOT / "opt/1_3B.txt", sep=" ", header=None, dtype=np.float32, skiprows=lambda i: i not in set(idx)).dropna(axis=1).to_numpy()

    # notebook recipe, verbatim from old/opt_squeeze.py.  transformers 4.20.1 predates the
    # current hub cache layout, so point it at a local snapshot directory holding
    # pytorch_model.bin (env OPT_PATH), else it downloads facebook/opt-1.3b afresh.
    import os
    src = os.environ.get("OPT_PATH", "facebook/opt-1.3b")
    tok = GPT2Tokenizer.from_pretrained(src)
    # REMOVE_FINAL_LN=1 loads with config._remove_final_layer_norm=True, the flag OPT configs of
    # that era used for checkpoints whose final LayerNorm was not to be applied.
    kw = {"_remove_final_layer_norm": True} if os.environ.get("REMOVE_FINAL_LN") == "1" else {}
    model = OPTModel.from_pretrained(src, **kw)
    model.eval()
    ln = model.decoder.final_layer_norm
    ln_info = {"final_layer_norm_present": ln is not None,
               "config._remove_final_layer_norm": getattr(model.config, "_remove_final_layer_norm", None),
               "config.do_layer_norm_before": getattr(model.config, "do_layer_norm_before", None)}
    if ln is not None:
        ln_info["gamma_mean"] = float(ln.weight.float().mean()); ln_info["gamma_std"] = float(ln.weight.float().std())
        ln_info["beta_abs_mean"] = float(ln.bias.float().abs().mean())
        # is it at init (weight==1, bias==0)?  That is what an unloaded LayerNorm looks like.
        ln_info["looks_uninitialised"] = bool(torch.allclose(ln.weight, torch.ones_like(ln.weight)) and torch.allclose(ln.bias, torch.zeros_like(ln.bias)))
    E = []
    for w in words:
        inputs = tok(w, return_tensors="pt")
        with torch.no_grad():
            out = model(**inputs)
        embeddings = out.last_hidden_state[0][1:]      # drop BOS, as opt_squeeze.py
        E.append(embeddings.mean(0).numpy())
    E = np.stack(E)
    nc, ne = np.linalg.norm(C, axis=1), np.linalg.norm(E, axis=1)
    cos = (C * E).sum(1) / (nc * ne)
    res = {"transformers": transformers.__version__, "torch": torch.__version__, "n_words": n, "words": words,
           "final_layer_norm": ln_info,
           "cos_vs_cache": {"median": float(np.median(cos)), "min": float(cos.min())},
           "norm_ratio_vs_cache": {"median": float(np.median(ne / nc)), "p5": float(np.percentile(ne / nc, 5)), "p95": float(np.percentile(ne / nc, 95))},
           "max_abs_diff_vs_cache": float(np.abs(C - E).max())}
    # compare with the current-transformers post-LN re-extraction if present
    p = ROOT / "results/cache/opt-1.3b_original_full_postln.npy"
    if not p.exists():
        p = ROOT / "results/cache/opt-1.3b_original_full.npy"
    if p.exists():
        R = np.load(p, mmap_mode="r")[idx]
        nr = np.linalg.norm(R, axis=1)
        res["vs_current_transformers_postln"] = {"cos_median": float(np.median((R * E).sum(1) / (nr * ne))),
                                                 "norm_ratio_median_current_over_420": float(np.median(nr / ne))}
    (ROOT / "results").mkdir(exist_ok=True)
    tag = "_noln" if kw else ""
    json.dump(res, open(ROOT / f"results/repro_tf420{tag}.json", "w"), indent=1)
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 20)
