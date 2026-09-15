"""Audit intrinsic-dimension and compression artifacts for one probe cell."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from .probe_compress import load_pruned, load_quantized, load_svd_state
from .probe_utils import atomic_json, load_config, read_table


UNDEFINED_REASON = "full_probe_has_no_positive_improvement"


def _audit_undefined_id(cell: Path) -> dict | None:
    """A cell whose full probe never beat its majority baseline has no d90 to audit."""
    sidecar = cell / "intrinsic_dimension.sidecar.json"
    if (cell / "intrinsic_dimension.parquet").exists() or (cell / "intrinsic_dimension.tsv").exists():
        return None
    if not sidecar.exists():
        return {"passed": False, "status": "missing",
                "errors": ["no intrinsic-dimension table and no sidecar"],
                "rows": 0, "artifacts_reloaded": 0, "d90": {}}
    recorded = json.loads(sidecar.read_text())
    if recorded.get("status") != "undefined" or recorded.get("reason") != UNDEFINED_REASON:
        return {"passed": False, "status": recorded.get("status"),
                "errors": [f"intrinsic-dimension table missing without a declared undefined outcome: "
                           f"status={recorded.get('status')!r} reason={recorded.get('reason')!r}"],
                "rows": 0, "artifacts_reloaded": 0, "d90": {}}
    stray = sorted(path.name for path in cell.glob("subspace.seed*.npz"))
    if stray:
        return {"passed": False, "status": "undefined",
                "errors": [f"undefined intrinsic dimension but subspace artifacts exist: {stray}"],
                "rows": 0, "artifacts_reloaded": 0, "d90": {}}
    return {"passed": True, "status": "undefined", "reason": UNDEFINED_REASON,
            "errors": [], "rows": 0, "artifacts_reloaded": 0, "d90": {},
            "full_tune_accuracy": recorded.get("full_tune_accuracy"),
            "majority_accuracy": recorded.get("majority_accuracy")}


def audit_id(cell: Path, cfg: dict) -> dict:
    undefined = _audit_undefined_id(cell)
    if undefined is not None:
        return undefined
    frame = read_table(cell / "intrinsic_dimension.parquet")
    expected = {int(seed) for seed in cfg["intrinsic_dimension"]["projection_seeds"]}
    observed = {int(seed) for seed in frame.projection_seed.unique()}
    errors: list[str] = []
    if observed != expected:
        errors.append(f"projection seeds differ: expected {sorted(expected)}, got {sorted(observed)}")
    hits = frame[frame.is_d90.astype(str).str.lower().isin(["true", "1"])]
    if set(hits.projection_seed.astype(int)) != expected or hits.groupby("projection_seed").size().ne(1).any():
        errors.append("each projection seed must have exactly one d90")
    artifacts = 0
    for row in frame.itertuples():
        path = cell / f"subspace.seed{int(row.projection_seed)}.d{int(row.dimension)}.npz"
        if not path.exists():
            errors.append(f"missing {path.name}")
            continue
        with np.load(path, allow_pickle=False) as saved:
            required = {"phi", "initialization_seed", "projection_seed", "dimension",
                        "full_parameter_count", "input_size", "hidden_size",
                        "standardizer_mean", "standardizer_scale"}
            missing = required.difference(saved.files)
            if missing:
                errors.append(f"{path.name} missing {sorted(missing)}")
                continue
            if int(saved["projection_seed"]) != int(row.projection_seed):
                errors.append(f"{path.name} projection seed mismatch")
            if int(saved["dimension"]) != int(row.dimension) or len(saved["phi"]) != int(row.dimension):
                errors.append(f"{path.name} dimension mismatch")
        artifacts += 1
    return {"passed": not errors, "status": "defined", "errors": errors, "rows": len(frame),
            "artifacts_reloaded": artifacts, "d90": {
                str(int(row.projection_seed)): int(row.dimension) for row in hits.itertuples()}}


def audit_compression(cell: Path, cfg: dict) -> dict:
    frame = read_table(cell / "compression.parquet")
    expected = {int(seed) for seed in cfg["seeds"]}
    observed = {int(seed) for seed in frame.seed.unique()}
    errors: list[str] = []
    if observed != expected:
        errors.append(f"probe seeds differ: expected {sorted(expected)}, got {sorted(observed)}")
    selected = frame[frame.selected.astype(str).str.lower().isin(["true", "1"])]
    if set(selected.seed.astype(int)) != expected or selected.groupby("seed").size().ne(1).any():
        errors.append("each probe seed must have exactly one selected artifact")
    if len(selected) and not selected.acceptable.astype(str).str.lower().isin(["true", "1"]).all():
        errors.append("a selected artifact is not acceptable")
    artifacts = 0
    for row in frame.itertuples():
        path = cell / "compression_seeds" / str(int(row.seed)) / row.artifact
        if not path.exists():
            errors.append(f"missing seed {row.seed}/{row.artifact}")
            continue
        if path.stat().st_size != int(row.bytes):
            errors.append(f"reported byte count differs for seed {row.seed}/{row.artifact}")
        try:
            if row.method == "quantization":
                load_quantized(path)
            elif row.method == "pruning":
                load_pruned(path)
            elif row.method == "svd":
                load_svd_state(path)
            else:
                errors.append(f"unknown compression method {row.method}")
                continue
        except Exception as exc:
            errors.append(f"cannot reload seed {row.seed}/{row.artifact}: {type(exc).__name__}: {exc}")
            continue
        artifacts += 1
    return {"passed": not errors, "errors": errors, "rows": len(frame),
            "artifacts_reloaded": artifacts,
            "selected": [{"seed": int(row.seed), "method": row.method,
                          "setting": float(row.setting), "bytes": int(row.bytes)}
                         for row in selected.itertuples()]}


def audit_cell(cell: Path, cfg: dict) -> dict:
    dimension = audit_id(cell, cfg)
    compression = audit_compression(cell, cfg)
    return {"cell": str(cell.resolve()), "passed": dimension["passed"] and compression["passed"],
            "intrinsic_dimension": dimension, "compression": compression}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--cell", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    result = audit_cell(Path(args.cell), load_config(args.config))
    atomic_json(result, args.output)
    if not result["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
