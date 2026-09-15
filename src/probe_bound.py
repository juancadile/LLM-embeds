"""E1: how large is the honest description of one decision rule?

Every rung of the ladder is a complete description of a map from *raw* frozen
activations to a label, so the rungs are comparable. A direct probe absorbs its
standardizer into the first affine layer, which is exact; the folded probe is
what gets compressed, and compression quality is judged on raw activations. A
subspace probe cannot absorb the standardizer, so its rung carries it.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .probe_models import Standardizer, load_probe, predict_logits, save_probe, state_vector
from .probe_utils import atomic_json, read_table, software_manifest, write_table


FLOAT32 = 4
STANDARDIZER_KEYS = ("standardizer_mean", "standardizer_scale")


def fold_standardizer(model, scaler: Standardizer):
    """Absorb standardization into the first affine layer.

    A standardized probe is w . ((x - m) / s) + b, which equals (w / s) . x plus
    a corrected bias. The rescaled probe consumes raw activations directly.
    """
    import torch
    import torch.nn as nn
    first = model if isinstance(model, nn.Linear) else model[0]
    mean = torch.as_tensor(scaler.mean, dtype=first.weight.dtype, device=first.weight.device)
    scale = torch.as_tensor(scaler.scale, dtype=first.weight.dtype, device=first.weight.device)
    with torch.no_grad():
        first.bias -= (first.weight @ (mean / scale))
        first.weight /= scale
    return model


def identity_standardizer(size: int) -> Standardizer:
    return Standardizer(np.zeros(size, dtype=np.float32), np.ones(size, dtype=np.float32))


def standardizer_is_absorbable(probe_path: Path, x: np.ndarray, tolerance: float = 1e-3) -> dict:
    """Verify the folding numerically rather than asserting it."""
    import copy
    model, scaler, _ = load_probe(probe_path)
    reference = predict_logits(model, scaler.transform(x))
    folded = fold_standardizer(copy.deepcopy(model), scaler)
    observed = predict_logits(folded, np.asarray(x, dtype=np.float32))
    difference = float(np.max(np.abs(reference - observed)))
    scale = float(np.max(np.abs(reference))) or 1.0
    return {"max_absolute_logit_difference": difference,
            "max_relative_logit_difference": difference / scale,
            "absorbable": difference / scale <= tolerance, "tolerance": tolerance}


def fold_probe(probe_path: Path, out_path: Path) -> Path:
    """Write a checkpoint of the folded probe with an identity standardizer."""
    model, scaler, metadata = load_probe(probe_path)
    folded = fold_standardizer(model, scaler)
    save_probe(out_path, folded, identity_standardizer(len(scaler.mean)),
               {**metadata, "standardizer": "folded_into_first_layer"})
    return out_path


def payload_bytes(artifact: Path, out: Path) -> int:
    """Serialized size of a folded artifact without its identity standardizer.

    Only an artifact produced from a folded probe may be measured this way: its
    stored standardizer is the identity, which carries no information. An
    artifact whose weights live in standardized-input space is refused, because
    dropping its standardizer would leave a description that cannot be applied
    to raw activations.
    """
    with np.load(artifact, allow_pickle=False) as saved:
        mean, scale = saved["standardizer_mean"], saved["standardizer_scale"]
        if len(mean) and not (np.all(mean == 0) and np.all(scale == 1)):
            raise ValueError(f"{artifact.name} was compressed in standardized-input space; "
                             "fold the standardizer before compressing")
        kept = {name: saved[name] for name in saved.files if name not in STANDARDIZER_KEYS}
    out.mkdir(parents=True, exist_ok=True)
    stripped = out / f"{artifact.stem}.payload.npz"
    np.savez(stripped, **kept)
    return int(stripped.stat().st_size)


def compress_folded(probe_path: Path, x_tune: np.ndarray, y_tune: np.ndarray,
                    x_test: np.ndarray, y_test: np.ndarray, cfg: dict, out: Path) -> pd.DataFrame:
    """Fold, then compress, then judge on raw activations with the declared rule."""
    from .probe_compress import run_compression
    out.mkdir(parents=True, exist_ok=True)
    folded = fold_probe(probe_path, out / "probe.folded.pt")
    frame = run_compression(folded, x_tune, y_tune, cfg, out, x_test=x_test, y_test=y_test)
    frame["payload_bytes"] = [payload_bytes(out / row.artifact, out / "payload")
                              for row in frame.itertuples()]
    return frame


def ladder(cell: Path, input_size: int, standardizer_bytes: int,
           subspace: dict[str, float], seeds: list[int],
           compressed: dict[str, dict] | None = None) -> pd.DataFrame:
    """Assemble every measured rung; sizes are bytes of a complete description.

    Direct probes are folded, so their rungs carry no standardizer. Subspace
    rungs reconstruct a probe over standardized inputs and therefore carry the
    standardizer, which the pipeline fits per predicate: `standalone_bytes` is
    the honest number. `bytes_if_standardizer_shared` records what the rung
    would cost under a design that declared one layer-level standardizer shared
    across predicates; no such design was run.
    """
    rows = []
    for kind in ("mlp", "linear"):
        probe = cell / f"probe.{kind}.seed{seeds[0]}.full.pt"
        if not probe.exists():
            continue
        model, _, _ = load_probe(probe)
        count = int(state_vector(model).size)
        rows.append({"rung": f"{kind}_float32", "probe": kind, "parameters": count,
                     "standalone_bytes": count * FLOAT32, "bytes_if_standardizer_shared": count * FLOAT32,
                     "evaluation_macro_f1": np.nan})
        measured = (compressed or {}).get(kind)
        if measured:
            rows.append({"rung": f"{kind}_compressed", "probe": kind, "parameters": count,
                         "standalone_bytes": measured["payload_bytes"],
                         "bytes_if_standardizer_shared": measured["payload_bytes"],
                         "evaluation_macro_f1": measured["evaluation_macro_f1"]})
        d90 = subspace.get(kind)
        if d90 and np.isfinite(d90):
            # theta0 and the projection regenerate from their seeds; phi does not.
            payload = int(d90) * FLOAT32 + 2 * FLOAT32
            rows.append({"rung": f"{kind}_subspace_d90", "probe": kind, "parameters": int(d90),
                         "standalone_bytes": payload + standardizer_bytes,
                         "bytes_if_standardizer_shared": payload, "evaluation_macro_f1": np.nan})
    return pd.DataFrame(rows)


def _selected(frame: pd.DataFrame) -> dict | None:
    """Median payload of the artifact the tuning rule selected, one per seed."""
    chosen = frame[frame.selected.astype(str).str.lower().isin(["true", "1"])]
    if not len(chosen):
        return None
    return {"payload_bytes": float(chosen.payload_bytes.median()), "seeds": int(len(chosen)),
            "stored_bytes": float(chosen.bytes.median()),
            "methods": sorted(set(chosen.method)), "settings": sorted(set(chosen.setting.astype(float))),
            "evaluation_macro_f1": float(chosen.test_macro_f1.mean())
            if "test_macro_f1" in chosen else float("nan")}


def run_bound_ladder(cfg: dict, root: Path, predicate: str, layer: str, out: Path) -> pd.DataFrame:
    from .probe_experiments import cell_dir, labels_with_rows, slug
    from .probe_id import run_id_cell
    primary = cfg["models"]["primary"]
    cell = cell_dir(root, primary, predicate, layer, "last")
    labels = labels_with_rows(root, primary, predicate)
    stable = labels[labels.stable.astype(str).str.lower().isin(["true", "1"])].copy()
    x = np.load(root / slug(primary) / f"activations.{layer}.last.npy", mmap_mode="r")
    rows_idx = stable.activation_row.astype(int).to_numpy()
    y = stable.label.astype(int).to_numpy()
    split = stable.split.to_numpy()
    tune, test = split == "tune", split == "evaluation"
    x_tune = np.asarray(x[rows_idx[tune]], dtype=np.float32)
    x_test = np.asarray(x[rows_idx[test]], dtype=np.float32)
    out.mkdir(parents=True, exist_ok=True)

    absorbed, compressed, tables = {}, {}, {}
    for kind in ("linear", "mlp"):
        first = cell / f"probe.{kind}.seed{cfg['seeds'][0]}.full.pt"
        if not first.exists():
            continue
        absorbed[kind] = standardizer_is_absorbable(first, x_test)
        results = []
        for seed in cfg["seeds"]:
            frame = compress_folded(cell / f"probe.{kind}.seed{seed}.full.pt", x_tune, y[tune],
                                    x_test, y[test], cfg["compression"],
                                    out / f"{kind}_folded_compression" / str(seed))
            frame["seed"] = seed
            results.append(frame)
        tables[kind] = pd.concat(results, ignore_index=True)
        write_table(tables[kind], out / f"compression.{kind}.folded.parquet")
        compressed[kind] = _selected(tables[kind])

    dimensions = run_id_cell(np.asarray(x), stable, cfg["probe"], cfg["intrinsic_dimension"],
                             out / "linear_intrinsic_dimension", kind="linear")
    hits = dimensions[dimensions.is_d90] if len(dimensions) and "is_d90" in dimensions else pd.DataFrame()
    linear_d90 = float(hits.dimension.median()) if len(hits) else float("nan")

    mlp_d90 = float("nan")
    mlp_dimensions = cell / "intrinsic_dimension.parquet"
    if mlp_dimensions.exists() or mlp_dimensions.with_suffix(".tsv").exists():
        frame = read_table(mlp_dimensions)
        mlp_hits = frame[frame.is_d90.astype(str).str.lower().isin(["true", "1"])]
        if len(mlp_hits):
            mlp_d90 = float(mlp_hits.dimension.median())

    table = ladder(cell, int(x.shape[1]), 2 * int(x.shape[1]) * FLOAT32,
                   {"linear": linear_d90, "mlp": mlp_d90}, list(cfg["seeds"]), compressed)
    write_table(table, out / "bound_ladder.parquet")
    atomic_json({**software_manifest(), "predicate": predicate, "layer": layer,
                 "input_size": int(x.shape[1]), "linear_d90": linear_d90, "mlp_d90": mlp_d90,
                 "procedure": "fold standardizer into first layer, compress the folded probe, "
                              "select on raw tuning activations, report on raw evaluation activations",
                 "standardizer_absorbable": absorbed,
                 "standardizer_bytes_if_counted": 2 * int(x.shape[1]) * FLOAT32,
                 "standardizer_scope": "fit per predicate on that predicate's stable coding rows; "
                                       "not shared across predicates in this pipeline",
                 "compressed_payloads": compressed},
                out / "bound_ladder.sidecar.json")
    return table


def main() -> None:
    import argparse
    from .probe_utils import load_config, output_dir
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--predicate", default="knows")
    parser.add_argument("--layer", required=True)
    parser.add_argument("--output", default=None)
    args = parser.parse_args()
    cfg = load_config(args.config)
    root = output_dir(cfg)
    out = Path(args.output) if args.output else root / "bound_ladder" / f"{args.predicate}.{args.layer}"
    print(run_bound_ladder(cfg, root, args.predicate, args.layer, out).to_string(index=False))


if __name__ == "__main__":
    main()
