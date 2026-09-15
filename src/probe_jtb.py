"""E2: is the model's KNOWS rule the conjunction of its own JTB component rules?

Components are turned into three numbers per example by cross-fitting over
independent scenario groups, so no example is scored by a probe that saw it or
any paraphrase of its scenario. The KNOWS labels are then coded online from
those numbers and compared against the full representation and two controls.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .probe_mdl import online_code, _example_bits
from .probe_models import Standardizer, predict_logits, train_probe
from .probe_utils import atomic_json, software_manifest, write_table


COMPONENTS = ("believes", "justified", "true", "lucky_guessed")


def crossfit_features(x: np.ndarray, rows: np.ndarray, labels: np.ndarray, split: np.ndarray,
                      groups: np.ndarray, probe_cfg: dict, seed: int, folds: int = 5) -> np.ndarray:
    """Out-of-fold probe outputs, with folds drawn over independent groups."""
    unique = np.unique(groups)
    rng = np.random.default_rng(seed)
    assignment = {group: index % folds for index, group in enumerate(rng.permutation(unique))}
    fold_of = np.array([assignment[g] for g in groups])
    out = np.full(len(rows), np.nan, dtype=np.float64)
    coding, tune = split == "coding", split == "tune"
    for fold in range(folds):
        held = fold_of == fold
        fit, early = coding & ~held, tune & ~held
        if len(np.unique(labels[fit])) < 2 or len(np.unique(labels[early])) < 2:
            raise ValueError(f"fold {fold} leaves a single-class training or tuning set")
        scaler = Standardizer.fit(x[rows[fit]])
        model = train_probe("linear", scaler.transform(x[rows[fit]]), labels[fit],
                            scaler.transform(x[rows[early]]), labels[early], probe_cfg, seed + fold)
        out[held] = predict_logits(model, scaler.transform(x[rows[held]]))
    if not np.isfinite(out).all():
        raise ValueError("cross-fitting left unscored examples")
    return out


def gettier_split_cost(y: np.ndarray, logits: np.ndarray, luck: np.ndarray,
                       groups: np.ndarray, seed: int, draws: int = 1000) -> dict:
    """Per-example coding cost on Gettier versus non-Gettier cases, with a grouped interval."""
    def mean_bits(mask):
        return _example_bits(y[mask], logits[mask]) / max(int(mask.sum()), 1)
    gettier, plain = luck == 1, luck == 0
    rng = np.random.default_rng(seed)
    unique = np.unique(groups)
    positions = {g: np.flatnonzero(groups == g) for g in unique}
    gaps = []
    for _ in range(draws):
        sampled = unique[rng.integers(0, len(unique), len(unique))]
        idx = np.concatenate([positions[g] for g in sampled])
        g_mask, p_mask = luck[idx] == 1, luck[idx] == 0
        if not g_mask.any() or not p_mask.any():
            continue
        gaps.append(_example_bits(y[idx][g_mask], logits[idx][g_mask]) / g_mask.sum()
                    - _example_bits(y[idx][p_mask], logits[idx][p_mask]) / p_mask.sum())
    return {"bits_per_label_gettier": mean_bits(gettier), "bits_per_label_plain": mean_bits(plain),
            "gap": mean_bits(gettier) - mean_bits(plain),
            "gap_ci_low": float(np.quantile(gaps, .025)) if gaps else float("nan"),
            "gap_ci_high": float(np.quantile(gaps, .975)) if gaps else float("nan"),
            "n_gettier": int(gettier.sum()), "n_plain": int(plain.sum()), "draws": len(gaps)}


def build_conditions(features: dict[str, np.ndarray], raw: np.ndarray,
                     shuffled: dict[str, np.ndarray], random3: np.ndarray) -> dict[str, np.ndarray]:
    """Declared condition set; subsets are named by their component initials."""
    def stack(*names):
        return np.column_stack([features[name] for name in names])
    conditions = {
        "jtb": stack("believes", "justified", "true"),
        "jtb_luck": stack("believes", "justified", "true", "lucky_guessed"),
        "believes": stack("believes"), "justified": stack("justified"),
        "true": stack("true"), "lucky_guessed": stack("lucky_guessed"),
        "believes_justified": stack("believes", "justified"),
        "believes_true": stack("believes", "true"),
        "justified_true": stack("justified", "true"),
        "raw_representation": raw,
        "random_three_dimensions": random3,
        "shuffled_components": np.column_stack(
            [shuffled[name] for name in ("believes", "justified", "true")]),
    }
    return conditions


def run_jtb_experiment(cfg: dict, root: Path, layer: str, out: Path, folds: int = 5) -> pd.DataFrame:
    from .probe_data import independent_groups
    from .probe_experiments import labels_with_rows, slug
    from .probe_mdl import group_permutation
    primary = cfg["models"]["primary"]
    target = "knows"
    frames = {p: labels_with_rows(root, primary, p).set_index("activation_row", drop=False)
              for p in (target, *COMPONENTS)}
    stable = {p: set(f.index[f.stable.astype(str).str.lower().isin(["true", "1"])])
              for p, f in frames.items()}
    common = np.array(sorted(set.intersection(*stable.values())), dtype=int)
    if len(common) < 500:
        raise ValueError(f"common cohort too small: {len(common)}")
    cohort = frames[target].loc[common]
    split = cohort.split.to_numpy()
    groups = independent_groups(cohort).to_numpy()
    y = cohort.label.astype(int).to_numpy()
    luck = frames["lucky_guessed"].loc[common].label.astype(int).to_numpy()
    x = np.load(root / slug(primary) / f"activations.{layer}.last.npy", mmap_mode="r")
    seed = int(cfg["seed"])

    features, shuffled = {}, {}
    for component in COMPONENTS:
        labels = frames[component].loc[common].label.astype(int).to_numpy()
        features[component] = crossfit_features(x, common, labels, split, groups,
                                                cfg["probe"], seed, folds)
        permuted = labels[group_permutation(groups, split, seed + 7)]
        shuffled[component] = crossfit_features(x, common, permuted, split, groups,
                                                cfg["probe"], seed + 11, folds)
    raw = np.asarray(x[common], dtype=np.float32)
    rng = np.random.default_rng(seed + 13)
    projection = rng.normal(size=(raw.shape[1], 3)) / np.sqrt(raw.shape[1])
    random3 = Standardizer.fit(raw).transform(raw) @ projection
    conditions = build_conditions(features, raw, shuffled, random3)

    coding, tune, evaluation = split == "coding", split == "tune", split == "evaluation"
    out.mkdir(parents=True, exist_ok=True)
    rows, gettier = [], {}
    for name, matrix in conditions.items():
        matrix = np.ascontiguousarray(matrix, dtype=np.float32)
        for kind in cfg["probe"]["kinds"]:
            for probe_seed in cfg["seeds"]:
                code, model, scaler = online_code(
                    matrix[coding], y[coding], matrix[tune], y[tune], groups[coding],
                    kind, cfg["probe"], int(probe_seed))
                logits = predict_logits(model, scaler.transform(matrix[evaluation]))
                rows.append({"condition": name, "features": matrix.shape[1], "probe": kind,
                             "seed": int(probe_seed), "bits_per_label": code["bits_per_label"],
                             "total_bits": code["total_bits"], "marginal_bits": code["marginal_bits"],
                             "compression": code["compression"],
                             "evaluation_bits_per_label": _example_bits(y[evaluation], logits) / int(evaluation.sum())})
                if name in ("jtb", "jtb_luck", "raw_representation") and int(probe_seed) == int(cfg["seeds"][0]):
                    gettier[f"{name}.{kind}"] = gettier_split_cost(
                        y[evaluation], logits, luck[evaluation], groups[evaluation], seed)
    frame = pd.DataFrame(rows)
    write_table(frame, out / "jtb_composition.parquet")
    atomic_json({**software_manifest(), "layer": layer, "target": target,
                 "cohort_size": int(len(common)), "folds": folds,
                 "cohort": "examples stable for knows and all four components",
                 "features": "out-of-fold linear probe logits, folds over independent scenario groups",
                 "gettier_analysis": gettier,
                 "split_counts": {k: int(v.sum()) for k, v in
                                  (("coding", coding), ("tune", tune), ("evaluation", evaluation))},
                 "class_counts": {"positive": int(y.sum()), "negative": int((1 - y).sum())},
                 "gettier_counts": {"gettier": int(luck.sum()), "plain": int((1 - luck).sum())}},
                out / "jtb_composition.sidecar.json")
    return frame


def main() -> None:
    import argparse
    from .probe_utils import load_config, output_dir
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--layer", required=True)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--output", default=None)
    args = parser.parse_args()
    cfg = load_config(args.config)
    root = output_dir(cfg)
    out = Path(args.output) if args.output else root / "jtb_composition" / args.layer
    frame = run_jtb_experiment(cfg, root, args.layer, out, folds=args.folds)
    summary = frame.groupby(["condition", "probe"]).bits_per_label.mean().sort_values()
    print(summary.to_string())


if __name__ == "__main__":
    main()
