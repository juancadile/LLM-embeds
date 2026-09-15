"""Constructive probe intrinsic dimension using an implicit CountSketch projection."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .probe_models import (Standardizer, binary_metrics, make_probe,
                           predict_logits, probe_device, train_probe)
from .probe_utils import atomic_json, seed_all, software_manifest, write_table


def projection_indices(total: int, dimension: int, seed: int):
    """Implicit orthonormal sparse columns, with no empty dimensions.

    At full dimension this is a signed permutation, hence full rank. The
    previous independent hash assignment left empty columns even at d = D.
    """
    import torch
    if not 1 <= dimension <= total:
        raise ValueError("dimension must be between one and the parameter count")
    generator = torch.Generator().manual_seed(seed)
    buckets = (torch.arange(total) % dimension)[torch.randperm(total, generator=generator)]
    signs = torch.randint(0, 2, (total,), generator=generator).float().mul_(2).sub_(1)
    signs /= torch.bincount(buckets, minlength=dimension)[buckets].float().sqrt()
    return buckets, signs


def _functional_logits(base, theta0, phi, buckets, signs, x):
    import torch
    from torch.func import functional_call
    vector = theta0 + signs * phi[buckets]
    params, offset = {}, 0
    for name, parameter in base.named_parameters():
        count = parameter.numel()
        params[name] = vector[offset:offset + count].view_as(parameter)
        offset += count
    return functional_call(base, params, (x,)).squeeze(-1)


def train_projected(x_train: np.ndarray, y_train: np.ndarray, x_valid: np.ndarray,
                    y_valid: np.ndarray, input_size: int, hidden_size: int,
                    dimension: int, cfg: dict, seed: int, kind: str = "mlp"):
    import torch
    from torch.utils.data import DataLoader, TensorDataset
    seed_all(int(cfg.get("initialization_seed", 2026)))
    device = probe_device(cfg)
    base = make_probe(kind, input_size, hidden_size).to(device)
    theta0 = torch.nn.utils.parameters_to_vector(base.parameters()).detach()
    total = theta0.numel()
    if dimension > total: raise ValueError("subspace exceeds full parameter count")
    generator = torch.Generator().manual_seed(seed)
    buckets, signs = projection_indices(total, dimension, seed)
    buckets, signs = buckets.to(device), signs.to(device)
    phi = torch.nn.Parameter(torch.zeros(dimension, device=device))
    optimizer = torch.optim.AdamW([phi], lr=float(cfg["learning_rate"]), weight_decay=float(cfg["weight_decay"]))
    loss_fn = torch.nn.BCEWithLogitsLoss()
    loader = DataLoader(TensorDataset(torch.from_numpy(x_train), torch.from_numpy(y_train.astype(np.float32))),
                        batch_size=int(cfg["batch_size"]), shuffle=True, generator=generator)
    best, best_loss, stale = None, float("inf"), 0
    for _ in range(int(cfg["epochs"])):
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss_fn(_functional_logits(base, theta0, phi, buckets, signs, xb), yb).backward()
            optimizer.step()
        with torch.inference_mode():
            logits = _functional_logits(base, theta0, phi, buckets, signs,
                                        torch.from_numpy(x_valid).to(device)).cpu().numpy()
        score = binary_metrics(y_valid, logits)["nll_nats"]
        if score < best_loss - 1e-6: best_loss, best, stale = score, phi.detach().clone(), 0
        else:
            stale += 1
            if stale >= int(cfg["patience"]): break
    phi = best
    def predict(x):
        with torch.inference_mode():
            return _functional_logits(base, theta0, phi, buckets, signs,
                                      torch.from_numpy(x).to(device)).cpu().numpy()
    predict.subspace_state = {"phi": phi.cpu().numpy(), "initialization_seed": int(cfg.get("initialization_seed", 2026)),
        "projection_seed": seed, "dimension": dimension, "full_parameter_count": total,
        "input_size": input_size, "hidden_size": hidden_size}
    return predict, total


def run_id_cell(x: np.ndarray, labels: pd.DataFrame, probe_cfg: dict, id_cfg: dict,
                out: Path, kind: str = "mlp") -> pd.DataFrame:
    stable = labels[labels["stable"]]
    rows_idx = stable["activation_row"].astype(int).to_numpy()
    y = stable["label"].astype(int).to_numpy()
    split = stable["split"].to_numpy()
    tr, va, te = split == "coding", split == "tune", split == "evaluation"
    scaler = Standardizer.fit(x[rows_idx[tr]])
    xs = scaler.transform(x[rows_idx])
    majority_label = int(y[tr].mean() >= .5)
    majority = float(np.mean(y[va] == majority_label))
    full = train_probe(kind, xs[tr], y[tr], xs[va], y[va], probe_cfg,
                       int(probe_cfg.get("initialization_seed", 2026)))
    full_tune = binary_metrics(y[va], predict_logits(full, xs[va]))["accuracy"]
    target = majority + float(id_cfg["target_fraction"]) * (full_tune - majority)
    if full_tune <= majority:
        out.mkdir(parents=True, exist_ok=True)
        atomic_json({**software_manifest(), "status": "undefined", "probe_kind": kind,
                     "reason": "full_probe_has_no_positive_improvement",
                     "full_tune_accuracy": full_tune, "majority_accuracy": majority,
                     "training_device": probe_device(probe_cfg),
                     "probe_config": probe_cfg, "intrinsic_dimension_config": id_cfg},
                    out / "intrinsic_dimension.sidecar.json")
        return pd.DataFrame()
    n_params = sum(p.numel() for p in full.parameters())
    dimensions = sorted(set(min(int(d), n_params) for d in id_cfg["dimensions"]))
    rows = []
    out.mkdir(parents=True, exist_ok=True)
    for seed in id_cfg["projection_seeds"]:
        for dimension in dimensions:
            predict, _ = train_projected(xs[tr], y[tr], xs[va], y[va], x.shape[1],
                                          int(probe_cfg["hidden_size"]), dimension, probe_cfg,
                                          int(seed), kind)
            np.savez(out / f"subspace.seed{seed}.d{dimension}.npz", **predict.subspace_state,
                     standardizer_mean=scaler.mean, standardizer_scale=scaler.scale)
            tune_metrics = binary_metrics(y[va], predict(xs[va]))
            # Test labels are not consulted until the tuning criterion selects d90.
            reached = tune_metrics["accuracy"] >= target
            test_metrics = binary_metrics(y[te], predict(xs[te])) if reached else {}
            rows.append({"projection_seed": seed, "dimension": dimension,
                         "full_parameter_count": n_params, "majority_accuracy": majority,
                         "full_tune_accuracy": full_tune, "target_accuracy": target,
                         **{f"tune_{k}": v for k, v in tune_metrics.items()},
                         **{f"test_{k}": v for k, v in test_metrics.items()}})
            if tune_metrics["accuracy"] >= target: break
    result = pd.DataFrame(rows)
    result["is_d90"] = False
    for seed, group in result.groupby("projection_seed"):
        hit = group[group["tune_accuracy"] >= group["target_accuracy"]]
        if len(hit): result.loc[hit.index[0], "is_d90"] = True
    out.mkdir(parents=True, exist_ok=True)
    write_table(result, out / "intrinsic_dimension.parquet")
    atomic_json({**software_manifest(), "probe_kind": kind,
                 "projection": "balanced sparse orthonormal columns", "parameterization": "theta0 + P phi",
                 "criterion": "smallest dimension reaching 90% of full-probe accuracy improvement over majority",
                 "projection_seeds": id_cfg["projection_seeds"], "full_parameter_count": n_params,
                 "training_device": probe_device(probe_cfg), "probe_config": probe_cfg,
                 "intrinsic_dimension_config": id_cfg},
                out / "intrinsic_dimension.sidecar.json")
    return result
