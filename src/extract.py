"""Extract word vectors from an open Hugging Face model with explicit, switchable choices.

layer   : "emb" (input embedding table output = hidden_states[0]), "p25", "p50", "p75", "final"
pooling : "mean" over the word's tokens, "last" token of the word
context : "alone"  -> the word by itself
          "prompt" -> "The word is: {w}"
BOS/EOS and any prompt tokens are never pooled; only tokens overlapping the word's
characters are.  Each configuration is saved as
results/cache/{model}_{layer}_{pool}_{ctx}.npy (rows aligned to the word list).
"""
from __future__ import annotations

import json
import re
import time
from pathlib import Path

import numpy as np

from .common import CACHE

LAYERS = ("emb", "p25", "p50", "p75", "final")
POOLS = ("mean", "last")
CONTEXTS = {"alone": "{w}", "prompt": "The word is: {w}"}


def slug(model_id: str) -> str:
    return re.sub(r"[^A-Za-z0-9.-]+", "-", model_id.split("/")[-1]).lower()


def cfg_name(model_id: str, layer: str, pool: str, ctx: str) -> str:
    return f"{slug(model_id)}_{layer}_{pool}_{ctx}"


def layer_indices(n_hidden: int) -> dict[str, int]:
    """n_hidden = number of hidden_states entries (embeddings + every block)."""
    last = n_hidden - 1
    return {"emb": 0, "p25": round(0.25 * last), "p50": round(0.5 * last), "p75": round(0.75 * last), "final": last}


def _word_span_mask(offsets, start, end):
    """Boolean mask of tokens whose char span overlaps [start, end)."""
    m = []
    for s, e in offsets:
        m.append(e > s and s < end and e > start)
    return np.array(m, dtype=bool)


def extract(model_id: str, words: list[str], out_prefix: str | None = None, batch_size: int = 64,
            dtype="bfloat16", device="cuda", layers=LAYERS, pools=POOLS, contexts=CONTEXTS,
            force: bool = False) -> dict:
    """Run every (layer, pool, ctx) configuration in one pass per context.
    Returns {cfg_name: path} plus token counts."""
    import torch
    from transformers import AutoModel, AutoTokenizer

    out_prefix = out_prefix or slug(model_id)
    CACHE.mkdir(parents=True, exist_ok=True)
    want = {(l, p, c) for l in layers for p in pools for c in contexts}
    paths = {}
    todo = set()
    for l, p, c in want:
        f = CACHE / f"{out_prefix}_{l}_{p}_{c}.npy"
        paths[(l, p, c)] = f
        stale = f.exists() and np.load(f, mmap_mode="r").shape[0] != len(words)
        if force or stale or not f.exists():
            todo.add((l, p, c))
    tok_path = CACHE / f"{out_prefix}_tokens.json"
    if not todo and tok_path.exists():
        return {"paths": {f"{out_prefix}_{l}_{p}_{c}": str(f) for (l, p, c), f in paths.items()},
                "tokens": json.load(open(tok_path)), "cached": True}

    tok = AutoTokenizer.from_pretrained(model_id)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "right"
    model = AutoModel.from_pretrained(model_id, dtype=getattr(torch, dtype)).to(device).eval()
    t0 = time.time()
    n_hidden = model.config.num_hidden_layers + 1
    lidx = layer_indices(n_hidden)
    tokens_per_word = {}
    for ctx, template in contexts.items():
        need_layers = sorted({lidx[l] for (l, p, c) in todo if c == ctx})
        if not need_layers:
            continue
        acc = {(l, p): np.zeros((len(words), model.config.hidden_size), dtype=np.float32)
               for (l, p, c) in todo if c == ctx}
        for b0 in range(0, len(words), batch_size):
            batch = words[b0:b0 + batch_size]
            texts = [template.format(w=w) for w in batch]
            enc = tok(texts, return_tensors="pt", padding=True, return_offsets_mapping=True, add_special_tokens=True)
            offsets = enc.pop("offset_mapping").numpy()
            with torch.no_grad():
                out = model(**{k: v.to(device) for k, v in enc.items()}, output_hidden_states=True)
            hs = {li: out.hidden_states[li].float().cpu().numpy() for li in need_layers}
            for r, (w, text) in enumerate(zip(batch, texts)):
                start = text.index(w) if ctx == "alone" else template.index("{w}")
                mask = _word_span_mask(offsets[r], start, start + len(w))
                if mask.sum() == 0:                       # fallback: all non-pad, non-special tokens
                    mask = enc["attention_mask"][r].numpy().astype(bool)
                    if hasattr(tok, "all_special_ids"):
                        ids = enc["input_ids"][r].numpy()
                        mask &= ~np.isin(ids, tok.all_special_ids)
                tokens_per_word.setdefault(ctx, {})[w] = int(mask.sum())
                pos = np.where(mask)[0]
                for (l, p), A in acc.items():
                    H = hs[lidx[l]][r]
                    A[b0 + r] = H[pos].mean(0) if p == "mean" else H[pos[-1]]
        for (l, p), A in acc.items():
            np.save(paths[(l, p, ctx)], A)
        print(f"[extract] {out_prefix} ctx={ctx} done ({time.time() - t0:.0f}s)", flush=True)
    if tokens_per_word:
        json.dump(tokens_per_word, open(tok_path, "w"))
    del model
    torch.cuda.empty_cache()
    return {"paths": {f"{out_prefix}_{l}_{p}_{c}": str(f) for (l, p, c), f in paths.items()},
            "tokens": tokens_per_word or (json.load(open(tok_path)) if tok_path.exists() else {}),
            "seconds": time.time() - t0, "n_hidden": n_hidden, "layer_indices": lidx}


def extract_original_recipe(model_id: str, words: list[str], out_prefix: str, device="cuda", dtype="bfloat16",
                            batch_size: int = 32) -> np.ndarray:
    """Re-create the notebooks' recipe for OPT / T5: word alone, final layer, mean over tokens.
    OPT: drop the leading BOS.  T5: encoder only, EOS *included* (as in old/t5.ipynb)."""
    import torch
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(model_id)
    is_t5 = "t5" in model_id.lower()
    if is_t5:
        from transformers import T5EncoderModel
        model = T5EncoderModel.from_pretrained(model_id, dtype=getattr(torch, dtype)).to(device).eval()
    else:
        from transformers import OPTModel
        model = OPTModel.from_pretrained(model_id, dtype=getattr(torch, dtype)).to(device).eval()
    out = np.zeros((len(words), model.config.hidden_size if not is_t5 else model.config.d_model), dtype=np.float32)
    tok.padding_side = "right"
    for b0 in range(0, len(words), batch_size):
        batch = words[b0:b0 + batch_size]
        enc = tok(batch, return_tensors="pt", padding=True)
        with torch.no_grad():
            H = model(**{k: v.to(device) for k, v in enc.items()}).last_hidden_state.float().cpu().numpy()
        am = enc["attention_mask"].numpy().astype(bool)
        for r in range(len(batch)):
            pos = np.where(am[r])[0]
            if not is_t5:
                pos = pos[1:]            # drop BOS (</s>) as opt_squeeze.py did
            out[b0 + r] = H[r, pos].mean(0)
    np.save(CACHE / f"{out_prefix}.npy", out)
    del model
    torch.cuda.empty_cache()
    return out
