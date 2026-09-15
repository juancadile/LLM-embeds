"""Hugging Face labeling and scenario-only activation extraction."""
from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import pandas as pd

from .probe_data import PREDICATES, QUESTIONS, assert_no_representation_leakage, combine_counterbalanced, forced_choice_prompt
from .probe_utils import atomic_json, seed_all, software_manifest, write_table, scenario_digest, config_digest


def _device(value: str) -> str:
    if value != "auto":
        return value
    import torch
    return "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")


def _load(model_id: str, dtype: str, device: str, revision: str):
    import torch
    import re
    from transformers import AutoModelForCausalLM, AutoTokenizer
    if not re.fullmatch(r"[0-9a-f]{40}", revision):
        raise ValueError("a pinned 40-character model commit is required")
    tok = AutoTokenizer.from_pretrained(model_id, revision=revision, use_fast=True, local_files_only=True)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "left"
    torch_dtype = getattr(torch, dtype) if device != "cpu" and hasattr(torch, dtype) else torch.float32
    model = AutoModelForCausalLM.from_pretrained(model_id, revision=revision,
        torch_dtype=torch_dtype, local_files_only=True).to(device).eval()
    model.requires_grad_(False)
    return tok, model


def _continuation_logprob(model, tok, prompt: str, answer: str, device: str) -> float:
    """Teacher-forced conditional log probability, valid for multi-token A/B strings."""
    return _continuation_logprobs(model, tok, [(prompt, answer)], device)[0]


def _continuation_logprobs(model, tok, requests: list[tuple[str, str]], device: str) -> list[float]:
    """Batched teacher-forced conditional log probabilities with left padding."""
    import torch
    encoded = []
    for prompt, answer in requests:
        prompt_ids = tok(prompt, add_special_tokens=True)["input_ids"]
        full_ids = tok(prompt + answer, add_special_tokens=True)["input_ids"]
        if full_ids[:len(prompt_ids)] != prompt_ids:
            # Tokenizers may merge at the boundary; a leading space makes choices stable.
            raise ValueError("answer tokenization did not preserve the prompt prefix")
        encoded.append((prompt_ids, full_ids))
    width = max(len(full_ids) for _, full_ids in encoded)
    pad = int(tok.pad_token_id)
    ids = torch.full((len(encoded), width), pad, dtype=torch.long, device=device)
    attention = torch.zeros_like(ids)
    answer_positions = []
    for row, (prompt_ids, full_ids) in enumerate(encoded):
        offset = width - len(full_ids)
        ids[row, offset:] = torch.tensor(full_ids, dtype=torch.long, device=device)
        attention[row, offset:] = 1
        answer_positions.append(range(offset + len(prompt_ids), width))
    position_ids = attention.cumsum(-1) - 1
    position_ids.masked_fill_(attention == 0, 0)
    with torch.inference_mode():
        logits = model(input_ids=ids, attention_mask=attention,
                       position_ids=position_ids).logits
        logp = torch.log_softmax(logits, dim=-1)
    scores = []
    for row, positions in enumerate(answer_positions):
        scores.append(float(sum(logp[row, i - 1, ids[row, i]].item() for i in positions)))
    return scores


def format_label_prompt(tok, row, predicate: str, reverse: bool, cfg: dict) -> str:
    raw = forced_choice_prompt(row, predicate, reverse)
    mode = cfg["labels"].get("format", "raw")
    variant = cfg["labels"].get("prompt_variant", "default")
    if variant == "explicit_mapping":
        question = QUESTIONS[predicate].format(name=row.name, claim=row.claim)
        if reverse:
            choices = "A. No\nB. Yes"
            mapping = "A means NO and B means YES."
        else:
            choices = "A. Yes\nB. No"
            mapping = "A means YES and B means NO."
        raw = ("Evaluate the scenario and answer the question. Return exactly one letter, A or B.\n"
               f"{mapping}\n\nScenario:\n{row.scenario_text}\n\nQuestion:\n{question}\n\n"
               f"Choices:\n{choices}\nResponse:")
    elif variant == "semantic_tokens":
        question = QUESTIONS[predicate].format(name=row.name, claim=row.claim)
        choices = "No\nYes" if reverse else "Yes\nNo"
        raw = ("Evaluate the scenario and answer the question. Choose one of the two answers "
               "listed below and return only that answer.\n\n"
               f"Scenario:\n{row.scenario_text}\n\nQuestion:\n{question}\n\n"
               f"Answer options:\n{choices}\nResponse:")
    if mode == "raw":
        return raw
    if mode != "chat":
        raise ValueError(f"unknown label prompt format: {mode}")
    return tok.apply_chat_template([{"role": "user", "content": raw}],
        tokenize=False, add_generation_prompt=True, enable_thinking=False)


def response_candidates(cfg: dict, reverse: bool) -> list[str]:
    """Candidate strings in displayed order for semantic normalization."""
    if cfg["labels"].get("prompt_variant", "default") == "semantic_tokens":
        return [" No", " Yes"] if reverse else [" Yes", " No"]
    return [" A", " B"]


def label_scenarios(frame: pd.DataFrame, model_id: str, cfg: dict, out: Path) -> pd.DataFrame:
    """Run both answer orders and retain only semantically stable high-margin judgments."""
    import torch
    seed_all(cfg["seed"])
    device = _device(cfg["extract"]["device"])
    revision = cfg["models"]["revisions"][model_id]
    tok, model = _load(model_id, cfg["extract"]["dtype"], device, revision)
    threshold = float(cfg["data"]["stable_probability"])
    rows = []
    started = time.time()
    for n, row in enumerate(frame.itertuples(index=False)):
        for predicate in cfg.get("predicates", PREDICATES):
            logits = []
            prompts = []
            for reverse in (False, True):
                prompt = format_label_prompt(tok, row, predicate, reverse, cfg)
                prompts.append(prompt)
            requests = [(prompts[reverse], answer)
                        for reverse in (False, True)
                        for answer in response_candidates(cfg, bool(reverse))]
            scores = _continuation_logprobs(model, tok, requests, device)
            logits = [scores[:2], scores[2:]]
            decision = combine_counterbalanced(logits[0], logits[1], threshold)
            rows.append({
                "example_id": row.example_id, "scenario_id": row.scenario_id,
                "paraphrase_id": row.paraphrase_id, "split": row.split,
                "predicate": predicate, "label": decision.label,
                "stable": decision.stable, "confidence": decision.confidence,
                "forward_label": decision.forward_label, "reversed_label": decision.reversed_label,
                "forward_margin": abs(logits[0][0] - logits[0][1]),
                "reversed_margin": abs(logits[1][0] - logits[1][1]),
            })
        if (n + 1) % 50 == 0:
            print(f"[labels] {model_id}: {n + 1}/{len(frame)}", flush=True)
    result = pd.DataFrame(rows)
    write_table(result, out / "labels.parquet")
    atomic_json({
        **software_manifest(), "model_id": model_id,
        "scenario_digest": scenario_digest(frame), "config_digest": config_digest(cfg),
        "requested_revision": revision, "tokenizer_revision": revision,
        "row_order": frame.example_id.tolist(),
        "model_revision": getattr(model.config, "_commit_hash", None),
        "prompt_version": cfg["labels"]["prompt_version"],
        "prompt_format": cfg["labels"].get("format", "raw"),
        "chat_template": tok.chat_template,
        "enable_thinking": False if cfg["labels"].get("format") == "chat" else None,
        "prompt_variant": cfg["labels"].get("prompt_variant", "default"),
        "prompt_template": "Read the scenario and select exactly A or B.\\n\\nScenario:\\n{scenario_text}\\n\\n{predicate_question}\\n{choices}\\nAnswer:",
        "predicate_questions": QUESTIONS,
        "answer_orders": [["Yes", "No"], ["No", "Yes"]],
        "scored_response_orders": [response_candidates(cfg, False), response_candidates(cfg, True)],
        "probability_threshold": threshold, "deterministic": True,
        "seconds": time.time() - started,
    }, out / "labels.sidecar.json")
    del model
    if torch.cuda.is_available(): torch.cuda.empty_cache()
    return result


def layer_indices(n_layers: int) -> dict[str, int]:
    return {"emb": 0, "p25": round(n_layers * .25), "p50": round(n_layers * .5),
            "p75": round(n_layers * .75), "final": n_layers}


def extract_scenarios(frame: pd.DataFrame, model_id: str, cfg: dict, out: Path) -> dict:
    """Extract activations from scenario text alone—never from the label prompt."""
    import torch
    seed_all(cfg["seed"])
    texts = frame["scenario_text"].astype(str).tolist()
    assert_no_representation_leakage(texts)
    device = _device(cfg["extract"]["device"])
    revision = cfg["models"]["revisions"][model_id]
    tok, model = _load(model_id, cfg["extract"]["dtype"], device, revision)
    base_layers = list(cfg["extract"]["layers"])
    layers = list(base_layers)
    pools = list(cfg["extract"]["pooling"])
    n_blocks = int(model.config.num_hidden_layers)
    indices = layer_indices(n_blocks)
    if cfg["extract"].get("store_all_layers_for_primary", False) and model_id == cfg["models"]["primary"]:
        block_indices = {f"block{i:02d}": i for i in range(1, n_blocks)}
        indices.update(block_indices)
        layers.extend(name for name in block_indices if name not in layers)
    hidden = int(model.config.hidden_size)
    out.mkdir(parents=True, exist_ok=True)
    analysis_pairs = [(layer, pool) for layer in layers
                      for pool in (pools if layer in base_layers else ["last"])]
    arrays = {
        (layer, pool): np.lib.format.open_memmap(
            out / f"activations.{layer}.{pool}.npy", mode="w+", dtype=np.float32,
            shape=(len(frame), hidden),
        ) for layer, pool in analysis_pairs
    }
    batch_size = int(cfg["extract"]["batch_size"])
    token_counts = np.zeros(len(frame), dtype=np.int32)
    extraction_positions = []
    for start in range(0, len(texts), batch_size):
        batch = texts[start:start + batch_size]
        enc = tok(batch, return_tensors="pt", padding=True, truncation=False,
                  return_special_tokens_mask=True)
        special = enc.pop("special_tokens_mask").numpy().astype(bool)
        cpu_mask = enc["attention_mask"].numpy().astype(bool) & ~special
        token_counts[start:start + len(batch)] = cpu_mask.sum(1)
        if enc["input_ids"].shape[1] > model.config.max_position_embeddings:
            raise ValueError("scenario exceeds context window; truncation is prohibited")
        position_ids = enc["attention_mask"].cumsum(-1) - 1
        position_ids.masked_fill_(enc["attention_mask"] == 0, 0)
        enc["position_ids"] = position_ids
        for r in range(len(batch)):
            positions = np.flatnonzero(cpu_mask[r])
            extraction_positions.append({"example_id": frame.iloc[start+r].example_id,
                "content_positions": position_ids[r, positions].tolist(),
                "content_token_ids": enc["input_ids"][r, positions].tolist()})
        with torch.inference_mode():
            result = model(**{k: v.to(device) for k, v in enc.items()}, output_hidden_states=True)
        for layer in layers:
            states = result.hidden_states[indices[layer]].float().cpu().numpy()
            for r in range(len(batch)):
                positions = np.flatnonzero(cpu_mask[r])
                if not len(positions): raise ValueError("scenario tokenized to no content tokens")
                for pool in (pools if layer in base_layers else ["last"]):
                    arrays[(layer, pool)][start + r] = (
                        states[r, positions[-1]] if pool == "last" else states[r, positions].mean(0)
                    )
        if start % (batch_size * 25) == 0:
            print(f"[extract] {model_id}: {min(start + batch_size, len(texts))}/{len(texts)}", flush=True)
    for arr in arrays.values(): arr.flush()
    np.save(out / "token_counts.npy", token_counts)
    question_controls = []
    if cfg["extract"].get("question_only_controls", True):
        for predicate in cfg.get("predicates", PREDICATES):
            question_texts = [QUESTIONS[predicate].format(name=row.name, claim=row.claim)
                              for row in frame.itertuples(index=False)]
            control_arrays = {
                (layer, pool): np.lib.format.open_memmap(
                    out / f"control.question_only.{predicate}.{layer}.{pool}.npy", mode="w+",
                    dtype=np.float32, shape=(len(frame), hidden))
                for layer in base_layers for pool in pools
            }
            for start in range(0, len(question_texts), batch_size):
                batch = question_texts[start:start + batch_size]
                enc = tok(batch, return_tensors="pt", padding=True, truncation=False,
                          return_special_tokens_mask=True)
                special = enc.pop("special_tokens_mask").numpy().astype(bool)
                mask = enc["attention_mask"].numpy().astype(bool) & ~special
                position_ids = enc["attention_mask"].cumsum(-1) - 1
                position_ids.masked_fill_(enc["attention_mask"] == 0, 0)
                enc["position_ids"] = position_ids
                with torch.inference_mode():
                    result = model(**{k: v.to(device) for k, v in enc.items()}, output_hidden_states=True)
                for layer in base_layers:
                    states = result.hidden_states[indices[layer]].float().cpu().numpy()
                    for r in range(len(batch)):
                        positions = np.flatnonzero(mask[r])
                        for pool in pools:
                            control_arrays[(layer, pool)][start + r] = (
                                states[r, positions[-1]] if pool == "last" else states[r, positions].mean(0))
            for arr in control_arrays.values(): arr.flush()
            question_controls.append(predicate)
    meta = {
        **software_manifest(), "model_id": model_id,
        "scenario_digest": scenario_digest(frame), "config_digest": config_digest(cfg),
        "requested_revision": revision, "tokenizer_revision": revision,
        "model_revision": getattr(model.config, "_commit_hash", None),
        "representation_input": "scenario_text column verbatim",
        "contains_question_or_answers": False, "n_examples": len(frame),
        "hidden_size": hidden, "layers": indices, "requested_layers": layers,
        "pooling": pools, "special_tokens_pooled": False,
        "question_only_controls": question_controls,
        "row_order": frame["example_id"].tolist(), "seed": cfg["seed"],
    }
    atomic_json(meta, out / "activations.sidecar.json")
    atomic_json(extraction_positions, out / "activations.positions.json")
    del model
    if torch.cuda.is_available(): torch.cuda.empty_cache()
    return meta
