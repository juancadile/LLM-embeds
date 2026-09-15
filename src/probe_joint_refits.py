"""Run all four estimators on paired scenario resamples for the fixed 15-cell grid."""
from __future__ import annotations

import json
from pathlib import Path
import numpy as np
import pandas as pd

from .probe_bootstrap import paired_refit_bootstrap, correlation_intervals, MEASURES
from .probe_utils import read_table, write_table, atomic_json, scenario_digest


def run_joint_refits(cfg: dict, root: Path, draws: int) -> dict:
    from .probe_experiments import labels_with_rows, cell_dir, slug
    from .probe_mdl import online_code
    from .probe_models import save_probe, binary_metrics, predict_logits
    from .probe_id import run_id_cell
    from .probe_compress import run_compression
    from .probe_llc import run_llc
    primary = cfg["models"]["primary"]
    calibration_path = root / slug(primary) / "llc_calibration" / "calibration.json"
    calibration = json.loads(calibration_path.read_text())
    if calibration["status"] != "accepted":
        raise ValueError("accepted frozen LLC calibration is required")
    best = calibration["probe_metadata"]["layer"]
    layers = ["emb", best, "final"]
    predicates = cfg["predicates"]
    if len(set(layers)) != 3 or len(predicates) != 5:
        raise ValueError("prespecified 15 distinct cells unavailable; no confirmatory correlations")
    labels = {p: labels_with_rows(root, primary, p).set_index("activation_row", drop=False)
              for p in predicates}
    common = sorted(set.intersection(*(set(frame.index[frame.stable]) for frame in labels.values())))
    minimum = int(cfg["thresholds"]["minimum_stable_per_class"])
    for predicate, frame in labels.items():
        counts = frame.loc[common].label.astype(int).value_counts().reindex([0, 1], fill_value=0)
        if counts.min() < minimum:
            raise ValueError(f"{predicate}: common stable cohort has insufficient class counts")
    scenarios = read_table(root / "scenarios.parquet").iloc[common].reset_index(drop=True)
    arrays = {layer: np.load(root / slug(primary) / f"activations.{layer}.last.npy", mmap_mode="r")
              for layer in layers}
    cells = [f"{predicate}/{layer}" for predicate in predicates for layer in layers]
    out = root / "joint_bootstrap"
    if out.exists():
        raise ValueError("joint bootstrap directory already exists; inspect it before another run")
    out.mkdir(parents=True)
    counter = 0

    def evaluate(cell, indices, chain_seed, copy_groups):
        nonlocal counter
        folder = out / f"fit{counter:06d}"
        counter += 1
        predicate, layer = cell.split("/")
        # ``indices`` are positions within the shared cohort, while labels
        # are keyed by their original activation rows.  Convert explicitly so
        # resampling remains correct even when activation rows are noncontiguous.
        selected = labels[predicate].loc[np.asarray(common, dtype=int)[indices]].reset_index(drop=True)
        selected["label"] = selected.label.astype(int)
        split = selected.split.to_numpy()
        coding, tune, test = split == "coding", split == "tune", split == "evaluation"
        # Degenerate resamples are retained as missing estimates by the driver.
        if any(selected.loc[mask, "label"].nunique() < 2 for mask in (coding, tune, test)):
            return {key: np.nan for key in MEASURES}
        x = arrays[layer]
        ids = selected.activation_row.to_numpy(dtype=int)
        y = selected.label.to_numpy(dtype=int)
        groups = np.asarray(copy_groups)
        code, model, scaler = online_code(x[ids[coding]], y[coding], x[ids[tune]], y[tune],
            groups[coding], "mlp", cfg["probe"], int(cfg["seeds"][0]))
        f1 = binary_metrics(y[test], predict_logits(model, scaler.transform(x[ids[test]])))["macro_f1"]
        probe = folder / "probe.pt"
        save_probe(probe, model, scaler, {"kind": "mlp", "input_size": x.shape[1],
            "hidden_size": cfg["probe"]["hidden_size"], "predicate": predicate,
            "layer": layer, "model_id": primary})
        dimensions = run_id_cell(x, selected, cfg["probe"], cfg["intrinsic_dimension"], folder)
        hits = dimensions[dimensions.is_d90] if "is_d90" in dimensions else pd.DataFrame()
        compression = run_compression(probe, x[ids[tune]], y[tune], cfg["compression"], folder,
            x_test=x[ids[test]], y_test=y[test])
        chosen = compression[compression.selected]
        chains = [chain_seed + i for i in range(int(calibration["settings"]["chains"]))]
        llc = run_llc(probe, x[ids[coding]], y[coding], calibration["settings"], chains, folder)
        central = llc[(llc.lr_multiplier == 1) & (llc.localization_multiplier == 1)]
        result = {"normalized_mdl": code["total_bits"] / code["marginal_bits"],
            "d90": float(hits.dimension.median()) if len(hits) == len(cfg["intrinsic_dimension"]["projection_seeds"]) else np.nan,
            "compressed_bytes": float(chosen.bytes.iloc[0]) if len(chosen) else np.nan,
            "llc": float(central.llc.mean()) if not llc.estimate_rejected.any() else np.nan}
        if f1 < cfg["thresholds"]["minimum_macro_f1"]:
            result = {key: np.nan for key in MEASURES}
        atomic_json({"cell_id": cell, "indices": indices, "metrics": result, "evaluation_macro_f1": f1}, folder / "refit.json")
        return result

    frame = paired_refit_bootstrap(scenarios, cells, evaluate, draws=draws, seed=int(cfg["seed"]))
    write_table(frame, out / "draws.parquet")
    summary = correlation_intervals(frame, cells)
    atomic_json({**summary, "cell_ids": cells, "draw_count": draws,
        "scenario_digest": scenario_digest(scenarios), "cohort": "intersection of stable examples across five predicates",
        "method": "paired_scenario_refits_with_independent_chains",
        "calibration_fingerprint": calibration["fingerprint"]}, out / "summary.json")
    return summary
