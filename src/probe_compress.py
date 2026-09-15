"""Reloadable probe compression formats with actual serialized byte counts."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from .probe_models import binary_metrics, load_probe, load_state_vector, predict_logits, state_vector
from .probe_utils import atomic_json, software_manifest, write_table


def _pack_codes(codes: np.ndarray, bits: int) -> bytes:
    codes = np.asarray(codes, dtype=np.uint64).ravel()
    planes = ((codes[:, None] >> np.arange(bits, dtype=np.uint64)) & 1).astype(np.uint8)
    return np.packbits(planes.reshape(-1), bitorder="little").tobytes()


def _unpack_codes(data: np.ndarray, count: int, bits: int) -> np.ndarray:
    raw = np.asarray(data, dtype=np.uint8)
    unpacked = np.unpackbits(raw, bitorder="little")[:count * bits].reshape(count, bits)
    return (unpacked.astype(np.int64) @ (1 << np.arange(bits, dtype=np.int64))).astype(np.int32)


def save_quantized(path: Path, vector: np.ndarray, bits: int, mean=None, std=None, metadata=None) -> None:
    limit = 2 ** (bits - 1) - 1
    scale = np.float32(max(np.max(np.abs(vector)) / max(limit, 1), 1e-12))
    signed = np.clip(np.rint(vector / scale), -limit, limit).astype(np.int32)
    codes = (signed + limit).astype(np.uint32)
    np.savez(path, format="symmetric_quant_v1", bits=np.int16(bits), count=np.int64(len(vector)),
             scale=scale, packed=np.frombuffer(_pack_codes(codes, bits), dtype=np.uint8),
             standardizer_mean=np.asarray([] if mean is None else mean, dtype=np.float32),
             standardizer_scale=np.asarray([] if std is None else std, dtype=np.float32),
             metadata=json.dumps(metadata or {}))


def load_quantized(path: Path) -> np.ndarray:
    with np.load(path, allow_pickle=False) as z:
        bits, count, scale = int(z["bits"]), int(z["count"]), float(z["scale"])
        limit = 2 ** (bits - 1) - 1
        return ((_unpack_codes(z["packed"], count, bits) - limit) * scale).astype(np.float32)


def save_pruned(path: Path, vector: np.ndarray, fraction: float, mean=None, std=None, metadata=None) -> None:
    keep = max(1, int(round(len(vector) * (1 - fraction))))
    indices = np.argpartition(np.abs(vector), -keep)[-keep:].astype(np.int32)
    order = np.argsort(indices); indices = indices[order]
    np.savez(path, format="magnitude_sparse_v1", count=np.int64(len(vector)),
             indices=indices, values=vector[indices].astype(np.float32),
             standardizer_mean=np.asarray([] if mean is None else mean, dtype=np.float32),
             standardizer_scale=np.asarray([] if std is None else std, dtype=np.float32),
             metadata=json.dumps(metadata or {}))


def load_pruned(path: Path) -> np.ndarray:
    with np.load(path, allow_pickle=False) as z:
        result = np.zeros(int(z["count"]), dtype=np.float32)
        result[z["indices"]] = z["values"]
        return result


def load_svd_state(path: Path) -> dict:
    """Reload a complete MLP state from the serialized input-matrix factors."""
    import torch
    with np.load(path, allow_pickle=False) as saved:
        if str(saved["format"]) != "input_svd_v1":
            raise ValueError("unknown SVD compression format")
        required = {"u", "s", "vt", "0.bias", "2.weight", "2.bias",
                    "standardizer_mean", "standardizer_scale", "metadata"}
        missing = required.difference(saved.files)
        if missing:
            raise ValueError(f"incomplete SVD artifact: {sorted(missing)}")
        state = {key: torch.from_numpy(saved[key].copy())
                 for key in ("0.bias", "2.weight", "2.bias")}
        state["0.weight"] = torch.from_numpy((saved["u"] * saved["s"]) @ saved["vt"])
        return {
            "state_dict": state,
            "standardizer_mean": saved["standardizer_mean"].copy(),
            "standardizer_scale": saved["standardizer_scale"].copy(),
            "metadata": json.loads(str(saved["metadata"])),
        }


def _evaluate_vector(model, vector, standardizer, x, y):
    load_state_vector(model, vector)
    return binary_metrics(y, predict_logits(model, standardizer.transform(x)))


def run_compression(probe_path: Path, x_eval: np.ndarray, y_eval: np.ndarray,
                    cfg: dict, out: Path, *, x_test: np.ndarray, y_test: np.ndarray) -> pd.DataFrame:
    """Select on tuning data (x_eval/y_eval); open test data only after selection."""
    model, scaler, metadata = load_probe(probe_path)
    original = state_vector(model).astype(np.float32)
    base = _evaluate_vector(model, original, scaler, x_eval, y_eval)
    out.mkdir(parents=True, exist_ok=True)
    original_path = out / "original.float32.npz"
    np.savez(original_path, format="dense_float32_v1", vector=original,
             standardizer_mean=scaler.mean, standardizer_scale=scaler.scale,
             metadata=json.dumps(metadata))
    rows = []
    for bits in cfg["quantization_bits"]:
        path = out / f"quantized.{bits}bit.npz"
        save_quantized(path, original, int(bits), scaler.mean, scaler.scale, metadata); restored = load_quantized(path)
        metrics = _evaluate_vector(model, restored, scaler, x_eval, y_eval)
        rows.append({"method": "quantization", "setting": bits, "bytes": path.stat().st_size,
                     "artifact": path.name, "retained_parameters": len(original), **metrics})
    for fraction in cfg["pruning"]:
        path = out / f"pruned.{int(float(fraction)*100)}pct.npz"
        save_pruned(path, original, float(fraction), scaler.mean, scaler.scale, metadata); restored = load_pruned(path)
        metrics = _evaluate_vector(model, restored, scaler, x_eval, y_eval)
        rows.append({"method": "pruning", "setting": fraction, "bytes": path.stat().st_size,
                     "artifact": path.name, "retained_parameters": int(np.count_nonzero(restored)), **metrics})
    # SVD is specific to the input-hidden matrix; all other tensors are serialized verbatim.
    # A single-layer probe has no such matrix: its weight is one row, so a low-rank
    # factorization stores more numbers than the weight itself. The move is skipped
    # rather than reported as an option that failed.
    load_state_vector(model, original)
    import torch.nn as nn
    if not isinstance(model, nn.Sequential):
        return _finalize(rows, original, original_path, base, cfg, out, model, scaler,
                         x_test, y_test, metadata)
    first = model[0].weight.detach().numpy()
    u, s, vt = np.linalg.svd(first, full_matrices=False)
    ranks = [2**p for p in range(int(np.floor(np.log2(min(first.shape)))) + 1)]
    state = {k: v.detach().numpy().copy() for k, v in model.state_dict().items() if k != "0.weight"}
    for rank in ranks:
        path = out / f"svd.rank{rank}.npz"
        payload = {"format": "input_svd_v1", "u": u[:, :rank].astype(np.float32),
                   "s": s[:rank].astype(np.float32), "vt": vt[:rank].astype(np.float32),
                   "standardizer_mean": scaler.mean, "standardizer_scale": scaler.scale,
                   "metadata": json.dumps(metadata), **state}
        np.savez(path, **payload)
        model.load_state_dict(load_svd_state(path)["state_dict"])
        metrics = binary_metrics(y_eval, predict_logits(model, scaler.transform(x_eval)))
        retained = rank * (first.shape[0] + first.shape[1] + 1) + sum(v.size for v in state.values())
        rows.append({"method": "svd", "setting": rank, "bytes": path.stat().st_size,
                     "artifact": path.name, "retained_parameters": retained, **metrics})
    return _finalize(rows, original, original_path, base, cfg, out, model, scaler,
                     x_test, y_test, metadata)


def _finalize(rows, original, original_path, base, cfg, out, model, scaler,
              x_test, y_test, metadata) -> pd.DataFrame:
    """Apply the acceptance rule, open test data only for the selected artifact, and record."""
    result = pd.DataFrame(rows)
    original_bytes = original_path.stat().st_size
    result["compression_ratio"] = original_bytes / result["bytes"]
    result["delta_nll"] = result["nll_nats"] - base["nll_nats"]
    result["accuracy_loss"] = base["accuracy"] - result["accuracy"]
    result["acceptable"] = ((result["delta_nll"] <= float(cfg["max_delta_nll"])) &
                            (result["accuracy_loss"] <= float(cfg["max_accuracy_loss"])))
    result["selection_split"] = "tune"
    result["selected"] = False
    selected_test = None
    if result.acceptable.any():
        chosen = result[result.acceptable].sort_values(["bytes", "artifact"]).index[0]
        result.loc[chosen, "selected"] = True
        row = result.loc[chosen]
        path = out / row.artifact
        if row.method == "quantization":
            load_state_vector(model, load_quantized(path))
        elif row.method == "pruning":
            load_state_vector(model, load_pruned(path))
        else:
            model.load_state_dict(load_svd_state(path)["state_dict"])
        selected_test = binary_metrics(y_test, predict_logits(model, scaler.transform(x_test)))
        base_test = _evaluate_vector(model, original, scaler, x_test, y_test)
        for key, value in selected_test.items():
            result.loc[chosen, f"test_{key}"] = value
        selected_test["delta_nll"] = selected_test["nll_nats"] - base_test["nll_nats"]
        selected_test["accuracy_loss"] = base_test["accuracy"] - selected_test["accuracy"]
    write_table(result, out / "compression.parquet")
    atomic_json({**software_manifest(), "base_metrics": base,
        "selection_split": "tune", "selected_test_metrics": selected_test,
        "original_parameter_bytes": original_bytes, "probe_metadata": metadata,
        "critical_compression_bytes": int(result.loc[result.acceptable, "bytes"].min()) if result.acceptable.any() else None,
        "compression_config": cfg}, out / "compression.sidecar.json")
    return result
