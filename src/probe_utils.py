"""Shared, dependency-light utilities for the model-relative probing study."""
from __future__ import annotations

import hashlib
import json
import os
import platform
import random
import subprocess
from pathlib import Path
from typing import Any

import numpy as np


def load_config(path: str | Path) -> dict[str, Any]:
    import yaml
    path = Path(path)
    cfg = yaml.safe_load(path.read_text())
    cfg["_config_path"] = str(path.resolve())
    return cfg


def seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        torch.use_deterministic_algorithms(True, warn_only=True)
    except ImportError:
        pass


def stable_int(text: str, seed: int = 0) -> int:
    return int.from_bytes(hashlib.sha256(f"{seed}:{text}".encode()).digest()[:8], "big")


def atomic_json(obj: Any, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2, default=json_default))
    os.replace(tmp, path)


def json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    raise TypeError(type(value).__name__)


def write_table(frame, path: str | Path) -> Path:
    """Write Parquet when available and an always-readable TSV companion."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path.with_suffix(".tsv"), sep="\t", index=False)
    try:
        frame.to_parquet(path, index=False)
        return path
    except (ImportError, ValueError):
        return path.with_suffix(".tsv")


def read_table(path: str | Path):
    import pandas as pd
    path = Path(path)
    if path.exists() and path.suffix == ".parquet":
        return pd.read_parquet(path)
    alt = path.with_suffix(".tsv")
    if alt.exists():
        return pd.read_csv(alt, sep="\t")
    raise FileNotFoundError(path)


def software_manifest() -> dict[str, Any]:
    versions = {"python": platform.python_version(), "numpy": np.__version__}
    for name in ("pandas", "torch", "transformers", "sklearn", "yaml"):
        try:
            mod = __import__(name)
            versions[name] = getattr(mod, "__version__", "unknown")
        except ImportError:
            versions[name] = None
    try:
        revision = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL
        ).strip()
    except Exception:
        revision = "unknown"
    source_hashes = {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                     for p in sorted(Path(__file__).parent.glob("probe_*.py"))}
    return {"software": versions, "git_revision": revision, "platform": platform.platform(),
            "probe_source_hashes": source_hashes}


def output_dir(cfg: dict[str, Any]) -> Path:
    return Path(cfg["output_dir"])


def scenario_digest(frame) -> str:
    """Fingerprint the actual ordered inputs and group assignments."""
    columns = [c for c in ("example_id", "scenario_id", "minimal_pair_group",
                           "split", "scenario_text") if c in frame]
    return hashlib.sha256(frame[columns].to_json(orient="records").encode()).hexdigest()


def config_digest(cfg: dict) -> str:
    values = {k: v for k, v in cfg.items() if not k.startswith("_")}
    return hashlib.sha256(json.dumps(values, sort_keys=True).encode()).hexdigest()


def config_diff(old: dict, new: dict, prefix: str = "") -> list[str]:
    """Dotted paths whose values differ between two configurations."""
    changed = []
    for key in sorted(set(old) | set(new)):
        path = f"{prefix}{key}"
        before, after = old.get(key, KeyError), new.get(key, KeyError)
        if isinstance(before, dict) and isinstance(after, dict):
            changed.extend(config_diff(before, after, f"{path}."))
        elif before != after:
            changed.append(path)
    return changed
