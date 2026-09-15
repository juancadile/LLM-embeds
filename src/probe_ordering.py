"""E3: does the cross-predicate codelength ordering survive scenario resampling?

Every predicate is refit on the same resampled index vector over the cohort that
is stable for all five, so the comparison is paired: a draw that happens to
contain easy scenarios makes every predicate look cheap at once.

Primary statistic is normalized MDL (total code over the marginal code), which
is comparable across predicates with different class balance. Raw bits per label
is recorded alongside it and is secondary.
"""
from __future__ import annotations

from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd

from .probe_bootstrap import paired_resample
from .probe_mdl import online_code
from .probe_utils import atomic_json, software_manifest, write_table


PREDICATES = ("believes", "justified", "knows", "true", "lucky_guessed")


def ordering_summary(frame: pd.DataFrame, statistic: str) -> dict:
    """Resampling probabilities for the ranking claims, with paired differences.

    When several probe seeds were run per draw, each (draw, predicate) cell is
    the mean over seeds before any ranking, so training variability enters the
    interval rather than being fixed at one seed.
    """
    wide = (frame.groupby(["draw", "predicate"])[statistic].mean()
                 .unstack("predicate").dropna())
    if wide.empty:
        raise ValueError("no complete draws")
    cheapest = wide.idxmin(axis=1).value_counts(normalize=True)
    dearest = wide.idxmax(axis=1).value_counts(normalize=True)
    pairs = {}
    for a, b in combinations(wide.columns, 2):
        difference = wide[a] - wide[b]
        pairs[f"{a}_minus_{b}"] = {
            "median": float(difference.median()),
            "ci_low": float(difference.quantile(.025)),
            "ci_high": float(difference.quantile(.975)),
            "probability_a_cheaper": float((difference < 0).mean()),
            "separated": bool(difference.quantile(.025) > 0 or difference.quantile(.975) < 0)}
    return {"statistic": statistic, "complete_draws": int(len(wide)),
            "probability_cheapest": {k: float(v) for k, v in cheapest.items()},
            "probability_dearest": {k: float(v) for k, v in dearest.items()},
            "per_predicate": {p: {"median": float(wide[p].median()),
                                  "ci_low": float(wide[p].quantile(.025)),
                                  "ci_high": float(wide[p].quantile(.975))} for p in wide.columns},
            "pairs": pairs}


def run_ordering_bootstrap(cfg: dict, root: Path, layer: str, out: Path, draws: int,
                           kind: str = "linear", seeds: list[int] | None = None) -> pd.DataFrame:
    from .probe_experiments import labels_with_rows, slug
    primary = cfg["models"]["primary"]
    frames = {p: labels_with_rows(root, primary, p).set_index("activation_row", drop=False)
              for p in PREDICATES}
    stable = {p: set(f.index[f.stable.astype(str).str.lower().isin(["true", "1"])])
              for p, f in frames.items()}
    common = np.array(sorted(set.intersection(*stable.values())), dtype=int)
    cohort = frames[PREDICATES[0]].loc[common]
    split = cohort.split.to_numpy()
    labels = {p: frames[p].loc[common].label.astype(int).to_numpy() for p in PREDICATES}
    x = np.asarray(np.load(root / slug(primary) / f"activations.{layer}.last.npy",
                           mmap_mode="r")[common], dtype=np.float32)
    seeds = [int(v) for v in (seeds if seeds is not None else cfg["seeds"])]
    out.mkdir(parents=True, exist_ok=True)
    rows, degenerate = [], 0
    for draw in range(draws):
        indices, d_groups = paired_resample(cohort, int(cfg["seed"]) + draw)
        d_split = split[indices]
        coding, tune = d_split == "coding", d_split == "tune"
        xd = x[indices]
        for predicate in PREDICATES:
            y = labels[predicate][indices]
            if min(len(np.unique(y[coding])), len(np.unique(y[tune]))) < 2:
                degenerate += 1
                rows.extend({"draw": draw, "predicate": predicate, "seed": seed,
                             "normalized_mdl": np.nan, "bits_per_label": np.nan} for seed in seeds)
                continue
            for seed in seeds:
                code, _, _ = online_code(xd[coding], y[coding], xd[tune], y[tune], d_groups[coding],
                                         kind, cfg["probe"], seed)
                rows.append({"draw": draw, "predicate": predicate, "seed": seed,
                             "normalized_mdl": code["total_bits"] / code["marginal_bits"],
                             "bits_per_label": code["bits_per_label"]})
        if (draw + 1) % 5 == 0:
            write_table(pd.DataFrame(rows), out / "ordering_draws.parquet")
    frame = pd.DataFrame(rows)
    write_table(frame, out / "ordering_draws.parquet")
    atomic_json({**software_manifest(), "layer": layer, "probe": kind, "draws": draws,
                 "seeds": seeds,
                 "aggregation": "per-draw mean over probe seeds before ranking"
                                if len(seeds) > 1 else "single probe seed; conditional on it",
                 "degenerate_cells": degenerate, "cohort_size": int(len(common)),
                 "cohort": "examples stable for all five predicates",
                 "resampling": "independent scenario groups within fixed splits, shared index vector, "
                               "each sampled copy labelled as its own group for the online coder",
                 "primary": ordering_summary(frame, "normalized_mdl"),
                 "secondary": ordering_summary(frame, "bits_per_label")},
                out / "ordering.sidecar.json")
    return frame


def main() -> None:
    import argparse
    from .probe_utils import load_config, output_dir
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--layer", required=True)
    parser.add_argument("--draws", type=int, required=True)
    parser.add_argument("--probe", default="linear")
    parser.add_argument("--seeds", default=None,
                        help="comma-separated probe seeds; default: every seed the protocol declares")
    parser.add_argument("--output", default=None)
    args = parser.parse_args()
    cfg = load_config(args.config)
    root = output_dir(cfg)
    out = Path(args.output) if args.output else root / "ordering_bootstrap" / args.layer
    seeds = [int(v) for v in args.seeds.split(",")] if args.seeds else None
    frame = run_ordering_bootstrap(cfg, root, args.layer, out, args.draws, args.probe, seeds)
    print(ordering_summary(frame, "normalized_mdl")["probability_cheapest"])


if __name__ == "__main__":
    main()
