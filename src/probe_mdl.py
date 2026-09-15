"""Online prequential coding following Voita & Titov (2020)."""
from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pandas as pd

from .probe_models import (Standardizer, binary_metrics, predict_logits,
                           probe_device, save_probe, train_probe)
from .probe_utils import atomic_json, software_manifest, write_table
from .probe_data import independent_groups, validate_grouped_splits


def group_permutation(groups: np.ndarray, splits: np.ndarray, seed: int) -> np.ndarray:
    """Exchange equal-size group vectors within splits, preserving dependence."""
    rng = np.random.default_rng(seed)
    permutation = np.arange(len(groups))
    for split in np.unique(splits):
        buckets = {}
        for group in np.unique(groups[splits == split]):
            positions = np.flatnonzero((groups == group) & (splits == split))
            buckets.setdefault(len(positions), []).append(positions)
        for positions in buckets.values():
            for target, source in zip(positions, rng.permutation(len(positions))):
                permutation[target] = positions[source]
    return permutation


def marginal_bits(y: np.ndarray) -> float:
    y = np.asarray(y, dtype=int)
    counts = np.bincount(y, minlength=2).astype(float)
    probs = np.maximum(counts / max(counts.sum(), 1), 1e-12)
    return float(-(counts * np.log2(probs)).sum())


def _example_bits(y: np.ndarray, logits: np.ndarray) -> float:
    logits = np.asarray(logits, dtype=np.float64)
    # Stable BCE in nats, converted to bits.
    nats = np.maximum(logits, 0) - logits * y + np.log1p(np.exp(-np.abs(logits)))
    return float(nats.sum() / math.log(2))


def endpoint_counts(n: int, fractions: list[float]) -> list[int]:
    values = sorted(set(max(1, min(n, int(math.ceil(n * f)))) for f in fractions))
    if values[-1] != n: values.append(n)
    return values


def grouped_bootstrap(y: np.ndarray, logits: np.ndarray, groups: np.ndarray,
                      seed: int, draws: int = 1000) -> dict[str, float]:
    rng = np.random.default_rng(seed)
    unique = np.unique(groups)
    values = {"nll_nats": [], "accuracy": [], "macro_f1": []}
    positions = {g: np.flatnonzero(groups == g) for g in unique}
    for _ in range(draws):
        sampled = unique[rng.integers(0, len(unique), len(unique))]
        idx = np.concatenate([positions[g] for g in sampled])
        metrics = binary_metrics(y[idx], logits[idx])
        for key in values: values[key].append(metrics[key])
    return {f"{key}_{bound}": float(np.quantile(vals, q))
            for key, vals in values.items() for bound, q in (("ci_low", .025), ("ci_high", .975))}


def online_code(x: np.ndarray, y: np.ndarray, x_tune: np.ndarray, y_tune: np.ndarray,
                groups: np.ndarray, kind: str, probe_cfg: dict, seed: int) -> tuple[dict, object, Standardizer]:
    """Encode group-shuffled coding data; each scaler sees only its training prefix."""
    rng = np.random.default_rng(seed)
    unique = np.unique(groups)
    shuffled = unique[rng.permutation(len(unique))]
    order = np.concatenate([np.flatnonzero(groups == g) for g in shuffled])
    x, y = x[order], y[order]
    # Percentages apply to independent groups, never individual paraphrases.
    group_ends = np.cumsum([np.count_nonzero(groups == g) for g in shuffled])
    endpoints = [int(group_ends[k - 1]) for k in
                 endpoint_counts(len(unique), probe_cfg["endpoints"])]
    first = endpoints[0]
    # The tuning set is shared side information, as it is for early stopping.
    # Reading future coding labels here would give the decoder unavailable data.
    tuning_counts = np.bincount(y_tune.astype(int), minlength=2).astype(float) + 0.5
    prior = tuning_counts / tuning_counts.sum()
    first_bits = float(-np.log2(prior[y[:first]]).sum())
    blocks = [{"train_end": 0, "test_start": 0, "test_end": first, "bits": first_bits,
               "code": "tuning_empirical_marginal", "prior": prior.tolist()}]
    model = scaler = None
    for left, right in zip(endpoints[:-1], endpoints[1:]):
        scaler = Standardizer.fit(x[:left])
        xt = scaler.transform(x[:left]); xv = scaler.transform(x_tune); xb = scaler.transform(x[left:right])
        model = train_probe(kind, xt, y[:left], xv, y_tune, probe_cfg, seed + left)
        logits = predict_logits(model, xb)
        blocks.append({"train_end": left, "test_start": left, "test_end": right,
                       "bits": _example_bits(y[left:right], logits), "code": "probe"})
    total = sum(b["bits"] for b in blocks)
    baseline = float(-np.log2(prior[y]).sum())
    # The evaluator receives the probe fitted on the complete coding split.
    # This fit is never used to price a block already encoded above.
    scaler = Standardizer.fit(x)
    model = train_probe(kind, scaler.transform(x), y, scaler.transform(x_tune),
                        y_tune, probe_cfg, seed)
    return ({"total_bits": total, "bits_per_label": total / len(y),
             "marginal_bits": baseline, "compression": 1 - total / baseline,
             "empirical_entropy_bits": marginal_bits(y),
             "conditioning": "tuning labels and representations shared with decoder",
             "group_order": shuffled.tolist(), "row_order": order.tolist(),
             "blocks": blocks, "n_labels": len(y)}, model, scaler)


def run_mdl_cell(x: np.ndarray, labels: pd.DataFrame, kind: str, probe_cfg: dict,
                 seeds: list[int], out: Path, metadata: dict) -> pd.DataFrame:
    stable = labels[labels["stable"]].copy()
    validate_grouped_splits(stable)
    idx = stable["activation_row"].astype(int).to_numpy()
    ys = stable["label"].astype(int).to_numpy()
    split = stable["split"].to_numpy()
    group = independent_groups(stable).to_numpy()
    tune, coding, evaluation = split == "tune", split == "coding", split == "evaluation"
    if len(np.unique(ys[tune])) < 2 or len(np.unique(ys[coding])) < 2:
        raise ValueError("tuning and coding splits must each contain both classes")
    rows = []
    out.mkdir(parents=True, exist_ok=True)
    for seed in seeds:
        code, model, scaler = online_code(x[idx[coding]], ys[coding], x[idx[tune]], ys[tune],
                                           group[coding], kind, probe_cfg, seed)
        eval_logits = predict_logits(model, scaler.transform(x[idx[evaluation]]))
        eval_metrics = binary_metrics(ys[evaluation], eval_logits)
        tune_metrics = binary_metrics(ys[tune], predict_logits(model, scaler.transform(x[idx[tune]])))
        intervals = grouped_bootstrap(ys[evaluation], eval_logits, group[evaluation], seed)
        rows.append({"seed": seed, "control": "observed", **{k: v for k, v in code.items() if k != "blocks"},
                     **eval_metrics, **{f"tune_{k}": v for k, v in tune_metrics.items()}, **intervals})
        pd.DataFrame({"activation_row": idx[evaluation], "scenario_id": group[evaluation],
                      "label": ys[evaluation], "logit": eval_logits}).to_csv(
                          out / f"evaluation_predictions.{kind}.seed{seed}.tsv", sep="\t", index=False)
        shuffled = ys[group_permutation(group, split, seed)]
        control, _, _ = online_code(x[idx[coding]], shuffled[coding], x[idx[tune]], shuffled[tune],
                                     group[coding], kind, probe_cfg, seed)
        rows.append({"seed": seed, "control": "shuffled_labels", **{k: v for k, v in control.items() if k != "blocks"}})
        perm = group_permutation(group, split, seed + 1)
        representation_control, _, _ = online_code(
            x[idx[perm][coding]], ys[coding], x[idx[perm][tune]], ys[tune], group[coding], kind, probe_cfg, seed)
        rows.append({"seed": seed, "control": "shuffled_representations",
                     **{k: v for k, v in representation_control.items() if k != "blocks"}})
        save_probe(out / f"probe.{kind}.seed{seed}.pt", model, scaler,
                   {**metadata, "kind": kind, "input_size": x.shape[1],
                    "hidden_size": int(probe_cfg["hidden_size"]), "seed": seed,
                    "training_device": probe_device(probe_cfg)})
        atomic_json({"observed": code, "shuffled_labels": control,
                     "shuffled_representations": representation_control}, out / f"online_code.{kind}.seed{seed}.json")
        # A separately named fully trained probe is the input to implementation compression.
        full_scaler = Standardizer.fit(x[idx[coding]])
        full_model = train_probe(kind, full_scaler.transform(x[idx[coding]]), ys[coding],
                                 full_scaler.transform(x[idx[tune]]), ys[tune], probe_cfg, seed)
        save_probe(out / f"probe.{kind}.seed{seed}.full.pt", full_model, full_scaler,
                   {**metadata, "kind": kind, "input_size": x.shape[1],
                    "hidden_size": int(probe_cfg["hidden_size"]), "seed": seed,
                    "training_device": probe_device(probe_cfg),
                    "training_data": "complete coding split; tuning split used for early stopping"})
    result = pd.DataFrame(rows)
    write_table(result, out / f"mdl.{kind}.parquet")
    atomic_json({**software_manifest(), "stage": "mdl", "cell": metadata,
                 "kind": kind, "seeds": seeds, "probe_config": probe_cfg,
                 "training_device": probe_device(probe_cfg)},
                out / f"mdl.{kind}.sidecar.json")
    return result
