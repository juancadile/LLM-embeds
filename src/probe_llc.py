"""Localized pSGLD learning-coefficient estimates and diagnostics.

The actual chains use the pinned DevInterp v2 low-level sampler because the
probe is a binary classifier, not an autoregressive language model.
"""
from __future__ import annotations

import importlib.metadata
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

from .probe_models import load_probe
from .probe_utils import atomic_json, seed_all, write_table


DEVINTERP_REVISION = "fbbf4c54e1f6ee46acb149f004df261fb05055c6"


def devinterp_provenance() -> dict:
    """Return and verify the exact DevInterp source used by the sampler."""
    try:
        distribution = importlib.metadata.distribution("devinterp")
    except importlib.metadata.PackageNotFoundError as exc:
        raise RuntimeError(
            "LLC requires the pinned DevInterp environment; install requirements-llc.txt"
        ) from exc
    direct = distribution.read_text("direct_url.json")
    direct_meta = json.loads(direct) if direct else {}
    actual = direct_meta.get("vcs_info", {}).get("commit_id")
    if actual != DEVINTERP_REVISION:
        raise RuntimeError(
            f"DevInterp revision mismatch: expected {DEVINTERP_REVISION}, got {actual or 'unrecorded'}"
        )
    return {
        "package": "devinterp",
        "version": distribution.version,
        "revision": actual,
        "url": direct_meta.get("url"),
    }


def trace_diagnostics(losses: np.ndarray) -> dict:
    """Split R-hat and initial-positive-sequence ESS for equal-length chains."""
    losses = np.asarray(losses, dtype=float)
    if losses.ndim != 2 or losses.shape[0] < 2 or losses.shape[1] < 20 or not np.isfinite(losses).all():
        return {"passed": False, "reason": "missing_or_nonfinite_traces"}
    half = losses.shape[1] // 2
    split = np.concatenate((losses[:, :half], losses[:, -half:]), axis=0)
    within = split.var(axis=1, ddof=1).mean()
    if within <= 0:
        return {"passed": False, "reason": "constant_traces"}
    between = half * split.mean(axis=1).var(ddof=1)
    variance = ((half - 1) * within + between) / half
    rhat = float(np.sqrt(variance / within))
    centered = split - split.mean(axis=1, keepdims=True)
    spectrum = np.fft.rfft(centered, n=2 * half, axis=1)
    autocov = np.fft.irfft(spectrum * spectrum.conjugate(), n=2 * half, axis=1)[:, :half] / half
    rho = 1 - (within - autocov.mean(axis=0)) / variance
    pair_sum = 0.0
    previous = float("inf")
    for lag in range(1, half - 1, 2):
        pair = min(previous, float(rho[lag] + rho[lag + 1]))
        if pair < 0:
            break
        pair_sum += pair
        previous = pair
    ess = float(min(split.size, split.size / max(1.0, 1 + 2 * pair_sum)))
    return {"passed": rhat <= 1.05 and ess >= 100, "rhat": rhat, "ess": ess,
            "reason": None if rhat <= 1.05 and ess >= 100 else "insufficient_mixing"}


def sensitivity_settings(multipliers: list[float]) -> tuple[tuple[float, float], ...]:
    values = {float(value) for value in multipliers}
    if not {0.5, 1.0, 2.0}.issubset(values):
        raise ValueError("LLC sensitivity_multipliers must contain 0.5, 1.0, and 2.0")
    return ((1., 1.), (.5, 1.), (2., 1.), (1., .5), (1., 2.))


def sensitivity_rejection(frame: pd.DataFrame, chains: int) -> str | None:
    """Require both axial neighbors of each calibrated hyperparameter."""
    means = {}
    for lr, gamma in sensitivity_settings([.5, 1., 2.]):
        cell = frame[(frame.lr_multiplier == lr) & (frame.localization_multiplier == gamma)]
        if len(cell) != chains or cell.chain.nunique() != chains:
            return "incomplete_sensitivity_grid"
        if not cell.accepted.all() or not np.isfinite(cell.llc).all():
            return "invalid_sensitivity_chains"
        means[(lr, gamma)] = cell.llc.mean()
    center = means[(1., 1.)]
    if center <= 0:
        return "non_positive_llc"
    if any(abs(value - center) / center > .10 for setting, value in means.items() if setting != (1., 1.)):
        return "sensitivity_exceeds_10_percent"
    return None


def _device(requested: str) -> str:
    import torch
    if requested == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("LLC device=cuda but CUDA is unavailable")
    if requested not in {"cpu", "cuda"}:
        raise ValueError("LLC device must be auto, cpu, or cuda")
    return requested


def divisor_batch_size(n: int, requested: int) -> int:
    """Avoid DevInterp's cycle/drop-last subset bias by covering every row."""
    if n <= 0 or requested <= 0:
        raise ValueError("dataset and batch sizes must be positive")
    for candidate in range(min(n, requested), 0, -1):
        if n % candidate == 0:
            return candidate
    raise AssertionError("one always divides a positive integer")


def _full_loss(model, x, y, device: str, batch_size: int) -> float:
    import torch
    total = 0.0
    was_training = model.training
    model.eval()
    with torch.inference_mode():
        for start in range(0, len(y), batch_size):
            xb = x[start:start + batch_size].to(device)
            yb = y[start:start + batch_size].to(device)
            loss = torch.nn.functional.binary_cross_entropy_with_logits(
                model(xb).squeeze(-1), yb, reduction="sum")
            total += float(loss)
    model.train(was_training)
    return total / len(y)


def psgld_chain(model, x: np.ndarray, y: np.ndarray, *, seed: int, learning_rate: float,
                localization: float, burn_in: int, draws: int, thinning: int,
                batch_size: int = 128, diagnostic_interval: int = 100,
                device: str = "auto") -> dict:
    """Run one pinned DevInterp RMSprop-SGLD chain for a BCE probe."""
    import torch
    from torch.utils.data import Dataset

    provenance = devinterp_provenance()
    from devinterp.optim import SGMCMC
    from devinterp.slt.sampler import sample_single_chain

    class ProbeDataset(Dataset):
        def __init__(self, features, targets):
            self.features = torch.from_numpy(np.asarray(features, dtype=np.float32))
            self.targets = torch.from_numpy(np.asarray(targets, dtype=np.float32))

        def __len__(self):
            return len(self.targets)

        def __getitem__(self, index):
            return {"input_ids": self.features[index], "labels": self.targets[index]}

    seed_all(seed)
    resolved_device = _device(device)
    x_t = torch.from_numpy(np.asarray(x, dtype=np.float32))
    y_t = torch.from_numpy(np.asarray(y, dtype=np.float32))
    n = len(y_t)
    effective_batch_size = divisor_batch_size(n, batch_size)
    n_beta = n / math.log(max(n, 3))
    reference = model.to(resolved_device).eval()
    trained_loss = _full_loss(reference, x_t, y_t, resolved_device, effective_batch_size)
    centers = {name: parameter.detach().to(resolved_device).clone()
               for name, parameter in reference.named_parameters()}
    losses: list[float] = []
    distances: list[float] = []
    diagnostic_draws: list[int] = []
    diagnostic_losses: list[float] = []
    transition_window: list[float] = []

    def evaluate(sampled_model, data):
        target = data["labels"].to(data["input_ids"].device)
        per_example = torch.nn.functional.binary_cross_entropy_with_logits(
            sampled_model(data["input_ids"]).squeeze(-1), target, reduction="none")
        return per_example, {}

    def micro_callback(*, loss, input_ids, chain, step, micro_step):
        del input_ids, chain, micro_step
        if step >= burn_in:
            transition_window.append(float(loss.mean()))

    def callback(*, loss, draw, chain, model, optimizer):
        del loss, chain, optimizer
        expected_window = 1 if draw == 0 else thinning
        if len(transition_window) != expected_window:
            raise RuntimeError(
                f"expected {expected_window} transition losses before draw, got {len(transition_window)}")
        losses.append(float(np.mean(transition_window)))
        transition_window.clear()
        squared = torch.zeros((), device=resolved_device)
        for name, parameter in model.named_parameters():
            squared.add_(((parameter - centers[name]) ** 2).sum())
        distances.append(float(squared.sqrt()))
        if draw % diagnostic_interval == 0 or draw == draws - 1:
            diagnostic_draws.append(int(draw))
            diagnostic_losses.append(_full_loss(model, x_t, y_t, resolved_device, effective_batch_size))

    try:
        sample_single_chain(
            ref_model=reference, dataset=ProbeDataset(x, y), evaluate=evaluate,
            param_masks={name: None for name, _ in reference.named_parameters()},
            num_draws=draws, num_burnin_steps=burn_in, num_steps_bw_draws=thinning,
            sampling_method=SGMCMC.rmsprop_sgld,
            sampling_method_kwargs={"lr": learning_rate, "localization": localization,
                                    "nbeta": n_beta},
            chain=0, seed=seed, dataloader_seed=seed, device=resolved_device,
            callbacks=[callback], micro_callback=micro_callback,
            batch_size=effective_batch_size, shuffle=True, epoch_mode="cycle")
    except Exception as exc:
        return {"accepted": False, "reason": f"sampler_error:{type(exc).__name__}",
                "llc": float("nan"), "loss_trace": losses,
                "distance_trace": distances, "diagnostic_draws": diagnostic_draws,
                "diagnostic_losses": diagnostic_losses, "devinterp": provenance}
    loss_array = np.asarray(losses, dtype=float)
    distance_array = np.asarray(distances, dtype=float)
    diagnostic_array = np.asarray(diagnostic_losses, dtype=float)
    finite = (len(loss_array) == draws and np.isfinite(loss_array).all() and
              np.isfinite(distance_array).all() and np.isfinite(diagnostic_array).all())
    llc = n_beta * (float(loss_array.mean()) - trained_loss) if finite else float("nan")
    return {
        "accepted": bool(finite and np.isfinite(llc)), "reason": None if finite else "incomplete_or_nonfinite_trace",
        "llc": llc, "trained_loss": trained_loss,
        "mean_sample_loss": float(loss_array.mean()) if len(loss_array) else float("nan"),
        "loss_sd": float(loss_array.std()) if len(loss_array) else float("nan"),
        "mean_distance": float(distance_array.mean()) if len(distance_array) else float("nan"),
        "max_distance": float(distance_array.max()) if len(distance_array) else float("nan"),
        "max_diagnostic_loss": float(diagnostic_array.max()) if len(diagnostic_array) else float("nan"),
        "loss_trace": losses, "distance_trace": distances,
        "diagnostic_draws": diagnostic_draws, "diagnostic_losses": diagnostic_losses,
        "n_beta": n_beta, "n": n, "device": resolved_device,
        "requested_batch_size": batch_size, "effective_batch_size": effective_batch_size,
        "devinterp": provenance,
    }


def run_llc(probe_path: Path, x: np.ndarray, y: np.ndarray, cfg: dict,
            seeds: list[int], out: Path) -> pd.DataFrame:
    if "max_basin_distance" not in cfg or "max_loss_increase" not in cfg:
        raise ValueError("LLC requires frozen calibration with basin distance and loss-increase limits")
    if len(set(seeds)) < int(cfg["chains"]):
        raise ValueError("each chain requires a distinct seed")
    provenance = devinterp_provenance()
    settings = sensitivity_settings(cfg["sensitivity_multipliers"])
    rows = []
    out.mkdir(parents=True, exist_ok=True)
    for lm, gm in settings:
        cell_rows, traces = [], []
        for chain, seed in enumerate(seeds[:int(cfg["chains"])]):
            model, scaler, _ = load_probe(probe_path)
            result = psgld_chain(
                model, scaler.transform(x), y, seed=seed,
                learning_rate=float(cfg["learning_rate"]) * lm,
                localization=float(cfg["localization"]) * gm,
                burn_in=int(cfg["burn_in"]), draws=int(cfg["draws"]),
                thinning=int(cfg["thinning"]), batch_size=int(cfg.get("batch_size", 128)),
                diagnostic_interval=int(cfg.get("diagnostic_interval", 100)),
                device=str(cfg.get("device", "auto")))
            losses = np.asarray(result.pop("loss_trace"))
            distances = np.asarray(result.pop("distance_trace"))
            diagnostic_draws = np.asarray(result.pop("diagnostic_draws"), dtype=np.int64)
            diagnostic_losses = np.asarray(result.pop("diagnostic_losses"))
            result.pop("devinterp", None)
            np.savez(out / f"trace.lr{lm}.gamma{gm}.chain{chain}.npz", losses=losses,
                     distances=distances, diagnostic_draws=diagnostic_draws,
                     diagnostic_losses=diagnostic_losses)
            if len(losses) != int(cfg["draws"]):
                result["accepted"] = False
                # A sampler exception or non-finite trace is the informative reason; keep it.
                result["reason"] = result.get("reason") or "incomplete_trace"
            elif (distances.max() > cfg["max_basin_distance"] or
                  result["max_diagnostic_loss"] - result["trained_loss"] > cfg["max_loss_increase"]):
                result.update(accepted=False, reason="left_trained_basin")
            traces.append(losses)
            cell_rows.append({"lr_multiplier": lm, "localization_multiplier": gm,
                              "chain": chain, "seed": seed, **result})
        diagnostics = trace_diagnostics(np.stack(traces)) if len({len(t) for t in traces}) == 1 else {"passed": False}
        for row in cell_rows:
            row["accepted"] = row["accepted"] and diagnostics["passed"]
            row.update({f"diagnostic_{key}": value for key, value in diagnostics.items()})
        rows.extend(cell_rows)
    frame = pd.DataFrame(rows)
    reason = sensitivity_rejection(frame, int(cfg["chains"]))
    frame["estimate_rejected"] = reason is not None
    frame["rejection_reason"] = reason
    write_table(frame, out / "llc.parquet")
    atomic_json({"sampler": "DevInterp SGMCMC.rmsprop_sgld", "loss": "binary_cross_entropy_with_logits",
                 "devinterp": provenance, "chains": cfg["chains"], "burn_in": cfg["burn_in"],
                 "draws": cfg["draws"], "thinning": cfg["thinning"],
                 "batch_size": int(cfg.get("batch_size", 128)),
                 "diagnostic_interval": int(cfg.get("diagnostic_interval", 100)),
                 "sensitivity_settings": settings, "rejected": reason is not None,
                 "rejection_reason": reason}, out / "llc.sidecar.json")
    return frame


def logistic_regular_expected_llc(parameter_count: int) -> float:
    """Reference value for a regular identifiable logistic model."""
    if parameter_count <= 0:
        raise ValueError("parameter_count must be positive")
    return parameter_count / 2


def validate_regular_logistic(seed: int = 2026) -> dict:
    """Numerically validate the executing DevInterp sampler against d/2."""
    import copy
    import torch
    seed_all(seed)
    rng = np.random.default_rng(seed)
    x = rng.normal(size=(300, 2)).astype(np.float32)
    probability = 1 / (1 + np.exp(-(x @ np.array([1.0, -1.0]) + .3)))
    y = (rng.random(len(x)) < probability).astype(np.float32)
    model = torch.nn.Linear(2, 1)
    xt, yt = torch.from_numpy(x), torch.from_numpy(y)
    optimizer = torch.optim.LBFGS(model.parameters(), max_iter=100, tolerance_grad=1e-9)

    def closure():
        optimizer.zero_grad()
        loss = torch.nn.functional.binary_cross_entropy_with_logits(model(xt).squeeze(-1), yt)
        loss.backward()
        return loss

    optimizer.step(closure)
    chains = [psgld_chain(copy.deepcopy(model), x, y, seed=seed + i,
        learning_rate=3e-3, localization=.01, burn_in=1000, draws=1000, thinning=10,
        batch_size=64, diagnostic_interval=100, device="auto") for i in range(4)]
    estimates = [chain["llc"] for chain in chains]
    diagnostics = trace_diagnostics(np.array([chain["loss_trace"] for chain in chains]))
    expected = logistic_regular_expected_llc(3)
    observed = float(np.mean(estimates))
    return {"parameter_count": 3, "expected_llc": expected, "observed_llc": observed,
            "chain_estimates": estimates, "relative_error": abs(observed - expected) / expected,
            "diagnostics": diagnostics, "devinterp": devinterp_provenance(),
            "passed": abs(observed - expected) / expected <= .20 and diagnostics["passed"]}
