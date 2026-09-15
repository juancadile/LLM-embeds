from pathlib import Path

import pandas as pd
import pytest

from src.probe_mdl_gate import summarize_mdl_layers


def _cell(root: Path, layer: str, observed_bits: float, f1: float):
    path = root / f"{layer}.last"
    path.mkdir(parents=True)
    rows = []
    for seed in (1, 2):
        rows.extend([
            {"seed": seed, "control": "observed", "total_bits": observed_bits,
             "marginal_bits": 100.0, "bits_per_label": observed_bits / 100, "macro_f1": f1},
            {"seed": seed, "control": "shuffled_labels", "total_bits": 100.0,
             "marginal_bits": 100.0, "bits_per_label": 1.0},
            {"seed": seed, "control": "shuffled_representations", "total_bits": 95.0,
             "marginal_bits": 100.0, "bits_per_label": 0.95},
        ])
    pd.DataFrame(rows).to_csv(path / "mdl.mlp.tsv", sep="\t", index=False)


def test_mdl_gate_selects_shortest_normalized_code_and_passes(tmp_path):
    _cell(tmp_path, "emb", 85.0, 0.80)
    _cell(tmp_path, "final", 70.0, 0.76)
    result = summarize_mdl_layers(tmp_path, ["emb", "final"], "last", 0.75, 0.10)
    assert result["selected_layer"] == "final"
    assert result["passed"] is True
    selected = result["layers"][1]
    assert selected["shuffled_label_improvement_mean"] == pytest.approx(0.30)
    assert selected["shuffled_representation_improvement_mean"] < 0.30


def test_mdl_gate_applies_thresholds_to_selected_layer(tmp_path):
    _cell(tmp_path, "emb", 85.0, 0.90)
    _cell(tmp_path, "final", 70.0, 0.70)
    result = summarize_mdl_layers(tmp_path, ["emb", "final"], "last", 0.75, 0.10)
    assert result["selected_layer"] == "final"
    assert result["passed"] is False
