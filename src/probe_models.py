"""Fixed probe architectures, prefix-local standardization, and metrics."""
from __future__ import annotations

import copy
from dataclasses import dataclass

import numpy as np


def make_probe(kind: str, input_size: int, hidden_size: int = 256):
    import torch.nn as nn
    if kind == "linear":
        return nn.Linear(input_size, 1)
    if kind == "mlp":
        return nn.Sequential(nn.Linear(input_size, hidden_size), nn.GELU(), nn.Linear(hidden_size, 1))
    raise ValueError(f"unknown probe kind: {kind}")


def probe_device(cfg: dict) -> str:
    """Resolve the probe-training device; inference activations remain frozen."""
    import torch
    requested = str(cfg.get("device", "auto"))
    if requested == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("probe.device=cuda but CUDA is unavailable")
    if requested not in {"cpu", "cuda"}:
        raise ValueError("probe.device must be auto, cpu, or cuda")
    return requested


@dataclass
class Standardizer:
    mean: np.ndarray
    scale: np.ndarray

    @classmethod
    def fit(cls, x: np.ndarray) -> "Standardizer":
        mean = np.asarray(x, dtype=np.float64).mean(0)
        scale = np.asarray(x, dtype=np.float64).std(0)
        scale[scale < 1e-8] = 1.0
        return cls(mean.astype(np.float32), scale.astype(np.float32))

    def transform(self, x: np.ndarray) -> np.ndarray:
        return ((np.asarray(x, dtype=np.float32) - self.mean) / self.scale).astype(np.float32)


def binary_metrics(y: np.ndarray, logits: np.ndarray, bins: int = 10) -> dict[str, float]:
    from sklearn.metrics import accuracy_score, f1_score
    y = np.asarray(y, dtype=np.int64)
    logits = np.asarray(logits, dtype=np.float64).reshape(-1)
    prob = 1 / (1 + np.exp(-np.clip(logits, -50, 50)))
    pred = (prob >= .5).astype(int)
    ece = 0.0
    edges = np.linspace(0, 1, bins + 1)
    for low, high in zip(edges[:-1], edges[1:]):
        mask = (prob >= low) & (prob < high if high < 1 else prob <= high)
        if mask.any():
            ece += mask.mean() * abs(y[mask].mean() - prob[mask].mean())
    return {
        "n": int(len(y)), "nll_nats": float(np.mean(np.logaddexp(0, logits) - y * logits)),
        "accuracy": float(accuracy_score(y, pred)),
        "macro_f1": float(f1_score(y, pred, labels=[0, 1], average="macro", zero_division=0)),
        "ece": float(ece),
    }


def predict_logits(model, x: np.ndarray, batch_size: int = 1024) -> np.ndarray:
    import torch
    model.eval()
    device = next(model.parameters()).device
    result = []
    with torch.inference_mode():
        for start in range(0, len(x), batch_size):
            batch = torch.from_numpy(np.asarray(
                x[start:start + batch_size], dtype=np.float32)).to(device)
            result.append(model(batch).squeeze(-1).cpu().numpy())
    return np.concatenate(result) if result else np.empty(0)


def train_probe(kind: str, x_train: np.ndarray, y_train: np.ndarray,
                x_valid: np.ndarray, y_valid: np.ndarray, cfg: dict, seed: int):
    import torch
    from torch.utils.data import DataLoader, TensorDataset
    torch.manual_seed(seed)
    device = probe_device(cfg)
    model = make_probe(kind, x_train.shape[1], int(cfg["hidden_size"])).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(cfg["learning_rate"]),
                                  weight_decay=float(cfg["weight_decay"]))
    loss_fn = torch.nn.BCEWithLogitsLoss()
    generator = torch.Generator().manual_seed(seed)
    loader = DataLoader(TensorDataset(torch.from_numpy(x_train), torch.from_numpy(y_train.astype(np.float32))),
                        batch_size=int(cfg["batch_size"]), shuffle=True, generator=generator)
    best, best_loss, stale = None, float("inf"), 0
    for _epoch in range(int(cfg["epochs"])):
        model.train()
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = loss_fn(model(xb).squeeze(-1), yb)
            loss.backward(); optimizer.step()
        valid_logits = predict_logits(model, x_valid)
        valid_loss = binary_metrics(y_valid, valid_logits)["nll_nats"]
        if valid_loss < best_loss - 1e-6:
            best_loss, best, stale = valid_loss, copy.deepcopy(model.state_dict()), 0
        else:
            stale += 1
            if stale >= int(cfg["patience"]): break
    model.load_state_dict(best)
    return model


def state_vector(model) -> np.ndarray:
    import torch
    return torch.nn.utils.parameters_to_vector(model.parameters()).detach().cpu().numpy()


def load_state_vector(model, vector: np.ndarray):
    import torch
    # vector_to_parameters installs views into its input tensor. Clone here so
    # later load_state_dict calls cannot mutate the supposedly immutable source
    # vector through aliased parameter storage.
    owned = torch.as_tensor(vector, dtype=torch.float32).clone()
    torch.nn.utils.vector_to_parameters(owned, model.parameters())
    return model


def save_probe(path, model, standardizer: Standardizer, metadata: dict) -> None:
    import torch
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": model.state_dict(), "mean": standardizer.mean,
                "scale": standardizer.scale, "metadata": metadata}, path)


def load_probe(path):
    import torch
    payload = torch.load(path, map_location="cpu", weights_only=False)
    meta = payload["metadata"]
    model = make_probe(meta["kind"], meta["input_size"], meta["hidden_size"])
    model.load_state_dict(payload["state_dict"])
    return model, Standardizer(payload["mean"], payload["scale"]), meta
