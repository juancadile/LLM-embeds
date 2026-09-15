"""Paired scenario bootstrap orchestration for complete complexity estimators.

An evaluator must refit MDL, d90, compression and LLC for each supplied resample.
Resampling an existing vector of cell scores is not scenario uncertainty.
"""
from __future__ import annotations

from collections.abc import Callable
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from .probe_data import independent_groups, validate_grouped_splits

MEASURES = ("normalized_mdl", "d90", "compressed_bytes", "llc")


def paired_resample(scenarios: pd.DataFrame, seed: int) -> tuple[np.ndarray, np.ndarray]:
    """Resample independent components within each fixed split.

    Every cell receives the same index vector, preserving the cross-predicate
    pairing and the original tuning/coding/evaluation separation.

    Returns the row indices and, aligned with them, a group label that is
    distinct for every sampled copy. A group drawn twice must enter the online
    coder as two groups: keeping the original label would merge the copies,
    shrink the number of groups the coder sees, and shift every prequential
    endpoint.
    """
    validate_grouped_splits(scenarios)
    groups = independent_groups(scenarios).to_numpy()
    split = scenarios.split.to_numpy()
    rng = np.random.default_rng(seed)
    indices, copies = [], []
    for name in sorted(np.unique(split)):
        keys = np.unique(groups[split == name])
        sampled = rng.choice(keys, len(keys), replace=True)
        for copy, key in enumerate(sampled):
            rows = np.flatnonzero((groups == key) & (split == name))
            indices.append(rows)
            copies.append(np.full(len(rows), f"{key}#{copy}", dtype=object))
    return np.concatenate(indices), np.concatenate(copies)


def paired_indices(scenarios: pd.DataFrame, seed: int) -> np.ndarray:
    return paired_resample(scenarios, seed)[0]


def paired_refit_bootstrap(scenarios: pd.DataFrame, cell_ids: list[str],
                          evaluate: Callable, *, draws: int, seed: int) -> pd.DataFrame:
    """Evaluate all cells on each common scenario resample.

    evaluate(cell_id, indices, chain_seed, groups) returns the four scalar
    estimators; ``groups`` labels every bootstrap copy distinctly.
    The chain seed varies independently by cell and draw; the scenario indices
    are paired. Invalid/censored estimates are retained as missing, not dropped.
    """
    if len(cell_ids) != 15 or len(set(cell_ids)) != 15:
        raise ValueError("the prespecified grid requires 15 distinct cells")
    if draws < 2:
        raise ValueError("at least two bootstrap draws are required")
    rng = np.random.default_rng(seed)
    rows = []
    for draw in range(draws):
        scenario_seed = int(rng.integers(0, 2**31))
        indices, copy_groups = paired_resample(scenarios, scenario_seed)
        for cell in cell_ids:
            chain_seed = int(rng.integers(0, 2**31))
            estimates = evaluate(cell, indices.copy(), chain_seed, copy_groups.copy())
            rows.append({"draw": draw, "cell_id": cell, "scenario_seed": scenario_seed,
                         "chain_seed": chain_seed,
                         **{key: estimates.get(key, np.nan) for key in MEASURES}})
    return pd.DataFrame(rows)


def correlation_intervals(draws: pd.DataFrame, cell_ids: list[str]) -> dict:
    """Refuse selective deletion of failed bootstrap cells or undefined d90."""
    if len(cell_ids) != 15 or len(set(cell_ids)) != 15:
        raise ValueError("expected exactly 15 prespecified cells")
    correlations = {key: [] for key in MEASURES[:-1]}
    for _, draw in draws.groupby("draw"):
        if len(draw) != 15 or set(draw.cell_id) != set(cell_ids) or draw.scenario_seed.nunique() != 1:
            return {"accepted": False, "reason": "unpaired_or_incomplete_draw"}
        if not np.isfinite(draw[list(MEASURES)].to_numpy(dtype=float)).all():
            return {"accepted": False, "reason": "invalid_or_censored_estimator"}
        for key in correlations:
            rho = float(spearmanr(draw.llc, draw[key]).statistic)
            if not np.isfinite(rho):
                return {"accepted": False, "reason": "constant_or_undefined_correlation"}
            correlations[key].append(rho)
    if any(len(values) < 1000 for values in correlations.values()):
        return {"accepted": False, "reason": "fewer_than_1000_paired_refits"}
    intervals = {key: {"low": float(np.quantile(values, .025)),
                       "median": float(np.median(values)),
                       "high": float(np.quantile(values, .975))}
                 for key, values in correlations.items()}
    return {"accepted": True, "intervals": intervals,
            "convergent": all(value["low"] > 0 for value in intervals.values())}
