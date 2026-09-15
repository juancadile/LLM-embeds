import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.probe_artifact_audit import audit_compression, audit_id
from src.probe_compress import save_pruned, save_quantized
from src.probe_utils import write_table


def test_id_audit_requires_all_seeds_hits_and_reloadable_states(tmp_path):
    cfg = {"intrinsic_dimension": {"projection_seeds": [0, 1]}}
    rows = []
    for seed, dimension in ((0, 2), (1, 4)):
        rows.append({"projection_seed": seed, "dimension": dimension, "is_d90": True})
        np.savez(tmp_path / f"subspace.seed{seed}.d{dimension}.npz",
                 phi=np.zeros(dimension), initialization_seed=9, projection_seed=seed,
                 dimension=dimension, full_parameter_count=10, input_size=2, hidden_size=3,
                 standardizer_mean=np.zeros(2), standardizer_scale=np.ones(2))
    write_table(pd.DataFrame(rows), tmp_path / "intrinsic_dimension.parquet")
    assert audit_id(tmp_path, cfg)["passed"]
    (tmp_path / "subspace.seed1.d4.npz").unlink()
    assert not audit_id(tmp_path, cfg)["passed"]


def test_id_audit_accepts_declared_undefined_dimension(tmp_path):
    cfg = {"intrinsic_dimension": {"projection_seeds": [0, 1]}}
    (tmp_path / "intrinsic_dimension.sidecar.json").write_text(json.dumps(
        {"status": "undefined", "reason": "full_probe_has_no_positive_improvement",
         "full_tune_accuracy": 0.71, "majority_accuracy": 0.72}))
    result = audit_id(tmp_path, cfg)
    assert result["passed"] and result["status"] == "undefined" and result["d90"] == {}


def test_id_audit_rejects_missing_table_without_declared_outcome(tmp_path):
    cfg = {"intrinsic_dimension": {"projection_seeds": [0, 1]}}
    assert not audit_id(tmp_path, cfg)["passed"]
    (tmp_path / "intrinsic_dimension.sidecar.json").write_text(json.dumps({"projection_seeds": [0, 1]}))
    assert not audit_id(tmp_path, cfg)["passed"]


def test_id_audit_rejects_undefined_cell_that_still_wrote_subspaces(tmp_path):
    cfg = {"intrinsic_dimension": {"projection_seeds": [0, 1]}}
    (tmp_path / "intrinsic_dimension.sidecar.json").write_text(json.dumps(
        {"status": "undefined", "reason": "full_probe_has_no_positive_improvement"}))
    np.savez(tmp_path / "subspace.seed0.d2.npz", phi=np.zeros(2))
    assert not audit_id(tmp_path, cfg)["passed"]


def test_compression_audit_checks_reported_bytes_and_selection(tmp_path):
    cfg = {"seeds": [11, 12]}
    rows = []
    for seed in cfg["seeds"]:
        folder = tmp_path / "compression_seeds" / str(seed)
        folder.mkdir(parents=True)
        path = folder / "quantized.8bit.npz"
        save_quantized(path, np.arange(10, dtype=np.float32), 8)
        rows.append({"seed": seed, "method": "quantization", "setting": 8,
                     "artifact": path.name, "bytes": path.stat().st_size,
                     "selected": True, "acceptable": True})
    write_table(pd.DataFrame(rows), tmp_path / "compression.parquet")
    assert audit_compression(tmp_path, cfg)["passed"]
    rows[0]["bytes"] += 1
    write_table(pd.DataFrame(rows), tmp_path / "compression.parquet")
    assert not audit_compression(tmp_path, cfg)["passed"]
