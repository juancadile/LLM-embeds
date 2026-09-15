"""Small deterministic CUDA smoke test for the fixed MLP probe optimizer."""
from __future__ import annotations

import json

import numpy as np

from src.probe_id import train_projected
from src.probe_models import binary_metrics, predict_logits, probe_device, train_probe


def main() -> None:
    rng = np.random.default_rng(11)
    x = rng.normal(size=(512, 32)).astype(np.float32)
    y = (x[:, :3] @ np.array([1.5, -1.0, 0.7], dtype=np.float32) > 0).astype(np.int64)
    train, valid = np.arange(400), np.arange(400, 512)
    cfg = {"hidden_size": 16, "learning_rate": 0.003, "weight_decay": 0.0001,
           "batch_size": 64, "epochs": 40, "patience": 8, "device": "cuda"}
    models = [train_probe("mlp", x[train], y[train], x[valid], y[valid], cfg, 123)
              for _ in range(2)]
    logits = [predict_logits(model, x[valid]) for model in models]
    metrics = binary_metrics(y[valid], logits[0])
    maximum_repeat_difference = float(np.max(np.abs(logits[0] - logits[1])))
    id_rng = np.random.default_rng(8)
    id_x = id_rng.normal(size=(240, 2)).astype(np.float32)
    id_y = (id_x[:, 0] + 0.2 * id_x[:, 1] > 0).astype(np.int64)
    id_cfg = {**cfg, "hidden_size": 4, "learning_rate": 0.02,
              "epochs": 20, "patience": 4}
    projected = [train_projected(id_x[:180], id_y[:180], id_x[180:], id_y[180:],
                                 2, 4, 16, id_cfg, 2)[0] for _ in range(2)]
    projected_logits = [predict(id_x[180:]) for predict in projected]
    projected_metrics = binary_metrics(id_y[180:], projected_logits[0])
    projected_repeat_difference = float(np.max(np.abs(
        projected_logits[0] - projected_logits[1])))
    result = {
        "device": probe_device(cfg),
        "macro_f1": metrics["macro_f1"],
        "accuracy": metrics["accuracy"],
        "maximum_repeat_difference": maximum_repeat_difference,
        "projected_accuracy": projected_metrics["accuracy"],
        "projected_maximum_repeat_difference": projected_repeat_difference,
        "passed": (metrics["macro_f1"] >= 0.90 and maximum_repeat_difference == 0.0 and
                   projected_metrics["accuracy"] >= 0.75 and
                   projected_repeat_difference == 0.0),
    }
    print(json.dumps(result, indent=2, sort_keys=True))
    if not result["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
