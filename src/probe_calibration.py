"""Calibrate once on the primary KNOWS/MDL-best cell and freeze the result."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

from .probe_llc import run_llc
from .probe_models import load_probe, state_vector
from .probe_utils import atomic_json


def calibration_fingerprint(probe: Path, x: np.ndarray, y: np.ndarray, cfg: dict) -> str:
    digest = hashlib.sha256(probe.read_bytes())
    digest.update(Path(__file__).read_bytes())
    digest.update(Path(__file__).with_name("probe_llc.py").read_bytes())
    for array in (x, y):
        array = np.ascontiguousarray(array)
        digest.update(str((array.shape, array.dtype.str)).encode())
        digest.update(array.tobytes())
    digest.update(json.dumps(cfg, sort_keys=True).encode())
    return digest.hexdigest()


def calibrate(probe: Path, x: np.ndarray, y: np.ndarray, cfg: dict,
              seeds: list[int], out: Path) -> dict:
    """Use a declared candidate grid; do not optimize against a desired LLC value.

    Select the first candidate (predeclared ordering) passing mixing, basin and
    all axial sensitivity checks. The result binds to the exact checkpoint,
    coding inputs, labels and calibration settings. Test data is never accepted.
    """
    fingerprint = calibration_fingerprint(probe, x, y, {**cfg, "chain_seeds": seeds})
    manifest = out / "calibration.json"
    if manifest.exists():
        frozen = json.loads(manifest.read_text())
        if frozen["fingerprint"] != fingerprint:
            raise ValueError("calibration inputs changed; use a new experiment directory")
        if frozen["status"] != "accepted":
            raise ValueError("no calibrated LLC setting passed; see calibration diagnostics")
        return frozen["settings"]
    model, _, metadata = load_probe(probe)
    if metadata.get("predicate") != "knows":
        raise ValueError("calibration must use KNOWS")
    norm = float(np.linalg.norm(state_vector(model)))
    limits = {
        "max_basin_distance": float(cfg.get("basin_relative_radius", .25)) * norm,
        "max_loss_increase": float(cfg.get("basin_loss_increase", .05)),
    }
    attempts = []
    for index, candidate in enumerate(cfg.get("calibration_candidates", [
            {"learning_rate": cfg["learning_rate"], "localization": cfg["localization"]}])):
        settings = {**cfg, **limits, **candidate}
        frame = run_llc(probe, x, y, settings, seeds, out / f"candidate{index:02d}")
        accepted = not bool(frame.estimate_rejected.any())
        attempts.append({"candidate": candidate, "accepted": accepted,
                         "reason": frame.rejection_reason.iloc[0]})
        if accepted:
            atomic_json({"status": "accepted", "fingerprint": fingerprint,
                "settings": settings, "attempts": attempts, "probe_metadata": metadata,
                "selection_data": "coding split only; MDL-best layer chosen from coding codelength"}, manifest)
            return settings
    atomic_json({"status": "rejected", "fingerprint": fingerprint, "attempts": attempts}, manifest)
    raise ValueError("no LLC calibration candidate passed mixing, basin, and sensitivity checks")
