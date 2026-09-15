"""Evaluate the preregistered primary KNOWS MDL gate."""
from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

import numpy as np

from .probe_utils import atomic_json, read_table, software_manifest


def summarize_mdl_layers(cells_root: Path, layers: list[str], pool: str,
                         minimum_macro_f1: float,
                         minimum_shuffled_improvement: float) -> dict:
    summaries = []
    for layer in layers:
        path = cells_root / f"{layer}.{pool}" / "mdl.mlp.parquet"
        frame = read_table(path)
        required = {"observed", "shuffled_labels", "shuffled_representations"}
        if set(frame.control) != required:
            raise ValueError(f"{path}: incomplete controls")
        indexed = {name: group.set_index("seed").sort_index()
                   for name, group in frame.groupby("control")}
        seeds = indexed["observed"].index
        if any(not seeds.equals(group.index) for group in indexed.values()):
            raise ValueError(f"{path}: control seed sets differ")
        observed = indexed["observed"]
        shuffled = indexed["shuffled_labels"]
        representations = indexed["shuffled_representations"]
        normalized = observed.total_bits / observed.marginal_bits
        shuffled_improvement = 1 - observed.total_bits / shuffled.total_bits
        representation_improvement = 1 - observed.total_bits / representations.total_bits
        macro_f1 = observed.macro_f1
        summaries.append({
            "layer": layer,
            "pool": pool,
            "seeds": [int(seed) for seed in seeds],
            "normalized_mdl_mean": float(normalized.mean()),
            "normalized_mdl_by_seed": [float(value) for value in normalized],
            "bits_per_label_mean": float(observed.bits_per_label.mean()),
            "macro_f1_mean": float(macro_f1.mean()),
            "macro_f1_by_seed": [float(value) for value in macro_f1],
            "shuffled_label_improvement_mean": float(shuffled_improvement.mean()),
            "shuffled_label_improvement_by_seed": [float(value) for value in shuffled_improvement],
            "shuffled_representation_improvement_mean": float(representation_improvement.mean()),
            "shuffled_representation_improvement_by_seed": [float(value) for value in representation_improvement],
        })
    best = min(summaries, key=lambda row: row["normalized_mdl_mean"])
    passed = (best["macro_f1_mean"] >= minimum_macro_f1 and
              best["shuffled_label_improvement_mean"] >= minimum_shuffled_improvement)
    return {
        "selection_rule": "lowest mean total_bits/marginal_bits among declared five-depth MLP cells",
        "gate_rule": "selected mean evaluation macro-F1 and paired-seed mean shuffled-label improvement",
        "minimum_macro_f1": float(minimum_macro_f1),
        "minimum_shuffled_label_improvement": float(minimum_shuffled_improvement),
        "selected_layer": best["layer"],
        "passed": bool(passed),
        "layers": summaries,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cells-root", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--layer", action="append", required=True)
    parser.add_argument("--pool", default="last")
    parser.add_argument("--minimum-macro-f1", type=float, required=True)
    parser.add_argument("--minimum-shuffled-improvement", type=float, required=True)
    args = parser.parse_args()
    result = summarize_mdl_layers(Path(args.cells_root), args.layer, args.pool,
                                  args.minimum_macro_f1,
                                  args.minimum_shuffled_improvement)
    result.update({
        "cells_root": str(Path(args.cells_root).resolve()),
        "gate_source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        **software_manifest(),
    })
    atomic_json(result, args.output)
    selected = next(row for row in result["layers"]
                    if row["layer"] == result["selected_layer"])
    print(f"selected={result['selected_layer']} normalized_mdl={selected['normalized_mdl_mean']:.6f} "
          f"macro_f1={selected['macro_f1_mean']:.6f} "
          f"shuffled_improvement={selected['shuffled_label_improvement_mean']:.6f} "
          f"passed={result['passed']}", flush=True)
    if not result["passed"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
