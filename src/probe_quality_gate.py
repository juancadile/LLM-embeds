"""Combine sample-count and final-layer F1 gates for comparative analyses."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from .probe_utils import atomic_json, read_table, software_manifest


def evaluate_quality_gate(model_root: Path, layers: list[str], pool: str,
                          minimum_macro_f1: float) -> dict:
    label_gate_path = model_root / "label_gate.json"
    label_gate = json.loads(label_gate_path.read_text())
    results = {}
    for predicate, sample in sorted(label_gate["predicates"].items()):
        layer_rows = []
        for layer in layers:
            path = model_root / "cells" / predicate / f"{layer}.{pool}" / "mdl.mlp.parquet"
            try:
                frame = read_table(path)
            except FileNotFoundError:
                continue
            observed = frame[frame.control == "observed"].set_index("seed").sort_index()
            shuffled = frame[frame.control == "shuffled_labels"].set_index("seed").sort_index()
            representations = frame[frame.control == "shuffled_representations"].set_index("seed").sort_index()
            if observed.empty or not observed.index.equals(shuffled.index) or not observed.index.equals(representations.index):
                raise ValueError(f"incomplete or unpaired MDL controls: {path}")
            layer_rows.append({
                "layer": layer,
                "normalized_mdl_mean": float((observed.total_bits / observed.marginal_bits).mean()),
                "macro_f1_mean": float(observed.macro_f1.mean()),
                "shuffled_label_improvement_mean": float(
                    (1 - observed.total_bits / shuffled.total_bits).mean()),
                "shuffled_representation_improvement_mean": float(
                    (1 - observed.total_bits / representations.total_bits).mean()),
                "seeds": [int(seed) for seed in observed.index],
            })
        final = next((row for row in layer_rows if row["layer"] == "final"), None)
        best = min(layer_rows, key=lambda row: row["normalized_mdl_mean"]) if layer_rows else None
        class_eligible = bool(sample["eligible"])
        f1_eligible = final is not None and final["macro_f1_mean"] >= minimum_macro_f1
        results[predicate] = {
            "class_eligible": class_eligible,
            "final_f1_eligible": bool(f1_eligible),
            "comparative_claim_eligible": bool(class_eligible and f1_eligible),
            "stable_negative": int(sample["negative"]),
            "stable_positive": int(sample["positive"]),
            "minimum_macro_f1": float(minimum_macro_f1),
            "final_layer": final,
            "mdl_best_layer": best["layer"] if best else None,
            "mdl_best": best,
            "layers": layer_rows,
        }
    return {"predicates": results}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-root", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--layer", action="append", required=True)
    parser.add_argument("--pool", default="last")
    parser.add_argument("--minimum-macro-f1", type=float, required=True)
    args = parser.parse_args()
    result = evaluate_quality_gate(Path(args.model_root), args.layer, args.pool,
                                   args.minimum_macro_f1)
    result.update({
        "model_root": str(Path(args.model_root).resolve()),
        "quality_gate_source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        **software_manifest(),
    })
    atomic_json(result, args.output)
    for predicate, row in result["predicates"].items():
        final_f1 = row["final_layer"]["macro_f1_mean"] if row["final_layer"] else None
        print(f"{predicate}: classes={row['class_eligible']} final_f1={final_f1} "
              f"eligible={row['comparative_claim_eligible']}", flush=True)


if __name__ == "__main__":
    main()
