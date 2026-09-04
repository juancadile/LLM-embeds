from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np

# torch 2.14 routes some matmuls through JIT-compiled triton kernels, which need Python
# headers that are not installed on the DGX (no root).  Disable before torch is imported.
os.environ.setdefault("TORCH_DISABLE_NATIVE_JIT", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

SEED = 2026
ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results"
CACHE = RESULTS / "cache"


def rng(offset: int = 0) -> np.random.Generator:
    return np.random.default_rng(SEED + offset)


def ensure_dirs() -> None:
    CACHE.mkdir(parents=True, exist_ok=True)


def dump_json(obj, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=1, default=_default)


def _default(o):
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    raise TypeError(type(o))


def fmt(x: float, nd: int = 4) -> str:
    return f"{x:.{nd}f}"


def md_table(header: list[str], rows: list[list], align_right_from: int = 1) -> str:
    """Render a GitHub-flavoured markdown table."""
    sep = []
    for i in range(len(header)):
        sep.append("---:" if i >= align_right_from else ":---")
    out = ["| " + " | ".join(str(h) for h in header) + " |", "| " + " | ".join(sep) + " |"]
    for r in rows:
        out.append("| " + " | ".join(str(c) for c in r) + " |")
    return "\n".join(out)


def env_note() -> str:
    import platform
    return f"python {platform.python_version()}, numpy {np.__version__}, host {platform.node()}"
