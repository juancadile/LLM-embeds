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


def hf_revision(model_id: str) -> str | None:
    """Commit hash of the cached snapshot, if the model is in the HF cache."""
    import os
    hub = Path(os.environ.get("HF_HOME", Path.home() / ".cache/huggingface")) / "hub" / f"models--{model_id.replace('/', '--')}"
    snaps = hub / "snapshots"
    if snaps.exists():
        revs = sorted(snaps.iterdir(), key=lambda p: p.stat().st_mtime)
        return revs[-1].name if revs else None
    return None


def write_sidecar(npy: Path, model_id: str, model, dtype: str, layer: str, layer_index: int, pool: str,
                  ctx: str, template: str, tokens: dict[str, int], n_vocab: int | None = None) -> None:
    """Metadata next to every configuration's .npy; a configuration without one is not done."""
    counts = list(tokens.values())
    n_rows = int(np.load(npy, mmap_mode="r").shape[0])
    layout = (f"{n_rows} rows = {n_vocab} vocab rows (vocab/expanded_vocab.txt, in order) + {n_rows - n_vocab} query-only rows "
              "(magnitude-pair words and define2 targets absent from the vocab; used as query vectors, never ranked or sampled)"
              if n_vocab else f"{n_rows} rows aligned to the word list passed to extract()")
    meta = {
        "row_layout": layout, "n_vocab_rows": n_vocab, "n_query_only_rows": (n_rows - n_vocab) if n_vocab else None,
        "model_id": model_id, "revision": hf_revision(model_id), "dtype": dtype,
        "architecture": (getattr(getattr(model, "config", None), "architectures", None) or [type(model).__name__])[0],
        "num_hidden_layers": getattr(getattr(model, "config", None), "num_hidden_layers", None),
        "hidden_size": getattr(getattr(model, "config", None), "hidden_size", None),
        "layer": layer, "layer_index": layer_index, "pooling": pool, "context": ctx, "context_template": template,
        "special_tokens_pooled": False,
        "n_words": n_rows,
        "tokens_per_word": {"mean": float(np.mean(counts)) if counts else None,
                            "max": int(max(counts)) if counts else None,
                            "frac_multi_token": float(np.mean([c > 1 for c in counts])) if counts else None},
        "written": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    json.dump(meta, open(npy.with_suffix(".json"), "w"), indent=1)


def backfill_sidecars(model_ids: list[str], dtype: str = "bfloat16", n_vocab: int | None = 5124, force: bool = False) -> list[str]:
    """Write missing sidecars for configurations extracted before sidecars existed."""
    from transformers import AutoConfig
    done = []
    for mid in model_ids:
        prefix = slug(mid)
        tok_path = CACHE / f"{prefix}_tokens.json"
        tokens = json.load(open(tok_path)) if tok_path.exists() else {}
        cfg = AutoConfig.from_pretrained(mid)
        lidx = layer_indices(cfg.num_hidden_layers + 1)

        class _M:  # minimal stand-in carrying the config
            config = cfg
        for f in CACHE.glob(f"{prefix}_*_*_*.npy"):
            if f.with_suffix(".json").exists() and not force:
                continue
            try:
                _, layer, pool, ctx = f.stem.rsplit("_", 3)
            except ValueError:
                continue
            if layer not in LAYERS or pool not in POOLS or ctx not in CONTEXTS:
                continue
            write_sidecar(f, mid, _M, dtype, layer, lidx[layer], pool, ctx, CONTEXTS[ctx], tokens.get(ctx, {}), n_vocab=n_vocab)
            done.append(f.stem)
    return done


def extract(model_id: str, words: list[str], out_prefix: str | None = None, batch_size: int = 64,
            dtype="bfloat16", device="cuda", layers=LAYERS, pools=POOLS, contexts=CONTEXTS,
            force: bool = False, n_vocab: int | None = None) -> dict:
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
            write_sidecar(paths[(l, p, ctx)], model_id, model, dtype, l, lidx[l], p, ctx, template, tokens_per_word.get(ctx, {}), n_vocab=n_vocab)
        print(f"[extract] {out_prefix} ctx={ctx} done ({time.time() - t0:.0f}s)", flush=True)
    if tokens_per_word:
        json.dump(tokens_per_word, open(tok_path, "w"))
    del model
    torch.cuda.empty_cache()
    return {"paths": {f"{out_prefix}_{l}_{p}_{c}": str(f) for (l, p, c), f in paths.items()},
            "tokens": tokens_per_word or (json.load(open(tok_path)) if tok_path.exists() else {}),
            "seconds": time.time() - t0, "n_hidden": n_hidden, "layer_indices": lidx}


def extract_original_recipe(model_id: str, words: list[str], out_prefix: str, device="cuda", dtype="bfloat16",
                            batch_size: int = 32) -> dict[str, np.ndarray]:
    """Re-create the notebooks' recipe for OPT / T5: word alone, final layer, mean over tokens.
    OPT: drop the leading BOS.  T5: encoder only, EOS *included* (as in old/t5.ipynb).

    Returns BOTH sides of the model's final normalisation layer, captured with a forward hook:
      "preln"  = decoder/encoder output before final_layer_norm (the residual stream),
      "postln" = after it (= last_hidden_state in current transformers).
    Saved as {out_prefix}_preln.npy and {out_prefix}_postln.npy.  The notebooks' cached vectors
    were produced by transformers 4.20.1, whose OPTModel did not load final_layer_norm, so which
    side the cache sits on is an empirical question answered by comparing against both."""
    import torch
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(model_id)
    is_t5 = "t5" in model_id.lower()
    if is_t5:
        from transformers import T5EncoderModel
        model = T5EncoderModel.from_pretrained(model_id, dtype=getattr(torch, dtype)).to(device).eval()
        ln = model.encoder.final_layer_norm
    else:
        from transformers import OPTModel
        model = OPTModel.from_pretrained(model_id, dtype=getattr(torch, dtype)).to(device).eval()
        ln = model.decoder.final_layer_norm
    captured = {}

    def hook(_m, inp, outp):
        captured["pre"] = inp[0].detach()
        captured["post"] = outp.detach()
    handle = ln.register_forward_hook(hook)
    d = model.config.hidden_size if not is_t5 else model.config.d_model
    out = {"preln": np.zeros((len(words), d), dtype=np.float32), "postln": np.zeros((len(words), d), dtype=np.float32)}
    tok.padding_side = "right"
    for b0 in range(0, len(words), batch_size):
        batch = words[b0:b0 + batch_size]
        enc = tok(batch, return_tensors="pt", padding=True)
        with torch.no_grad():
            model(**{k: v.to(device) for k, v in enc.items()})
        am = enc["attention_mask"].numpy().astype(bool)
        Hs = {"preln": captured["pre"].float().cpu().numpy(), "postln": captured["post"].float().cpu().numpy()}
        for r in range(len(batch)):
            pos = np.where(am[r])[0]
            if not is_t5:
                pos = pos[1:]            # drop BOS (</s>) as opt_squeeze.py did
            for k in out:
                out[k][b0 + r] = Hs[k][r, pos].mean(0)
    handle.remove()
    for k, A in out.items():
        np.save(CACHE / f"{out_prefix}_{k}.npy", A)
    del model
    torch.cuda.empty_cache()
    return out
