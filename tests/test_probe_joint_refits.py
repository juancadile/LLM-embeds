"""Integration checks for wiring; sampler accuracy is tested separately."""
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd

from src.probe_joint_refits import run_joint_refits
from src.probe_models import make_probe, Standardizer
from src.probe_utils import atomic_json, write_table


class JointRefitIntegration(unittest.TestCase):
    def test_all_estimators_are_called_for_every_paired_cell(self):
        n = 120
        frame = pd.DataFrame({"activation_row": np.arange(n), "example_id": [f"e{i}" for i in range(n)],
            "scenario_id": np.repeat(np.arange(40), 3).astype(str),
            "split": ["coding"]*72 + ["tune"]*24 + ["evaluation"]*24,
            "stable": True, "label": np.repeat(np.arange(40) % 2, 3)})
        cfg = {"models": {"primary": "test"}, "predicates": ["knows", "believes", "true", "justified", "lucky_guessed"],
            "seeds": [1], "seed": 2, "thresholds": {"minimum_stable_per_class": 1, "minimum_macro_f1": .75},
            "probe": {"hidden_size": 4}, "intrinsic_dimension": {"projection_seeds": [1]}, "compression": {}}
        x = np.random.default_rng(1).normal(size=(n, 2)).astype(np.float32)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            write_table(frame, root / "scenarios.parquet")
            (root / "test").mkdir(exist_ok=True)
            for layer in ("emb", "p50", "final"):
                np.save(root / "test" / f"activations.{layer}.last.npy", x)
            atomic_json({"status": "accepted", "probe_metadata": {"layer": "p50"},
                "settings": {"chains": 2}, "fingerprint": "test-only"}, root / "test/llc_calibration/calibration.json")
            model = make_probe("mlp", 2, 4)
            with patch("src.probe_experiments.labels_with_rows", return_value=frame), \
                 patch("src.probe_mdl.online_code", return_value=({"total_bits": 10, "marginal_bits": 20}, model, Standardizer.fit(x))) as mdl, \
                 patch("src.probe_models.binary_metrics", return_value={"macro_f1": 1}), \
                 patch("src.probe_id.run_id_cell", return_value=pd.DataFrame({"is_d90": [True], "dimension": [1]})) as dimension, \
                 patch("src.probe_compress.run_compression", return_value=pd.DataFrame({"selected": [True], "bytes": [100]})) as compression, \
                 patch("src.probe_llc.run_llc", return_value=pd.DataFrame({"lr_multiplier": [1, 1], "localization_multiplier": [1, 1],
                     "llc": [2., 3.], "estimate_rejected": [False, False]})) as llc:
                result = run_joint_refits(cfg, root, draws=2)
                self.assertFalse(result["accepted"])
                self.assertEqual([mdl.call_count, dimension.call_count, compression.call_count, llc.call_count], [30]*4)


if __name__ == "__main__": unittest.main()
