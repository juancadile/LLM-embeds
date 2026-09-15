import json
from pathlib import Path

import pandas as pd

from src.probe_quality_gate import evaluate_quality_gate


def _mdl(model_root: Path, predicate: str, layer: str, bits: float, f1: float):
    path = model_root / "cells" / predicate / f"{layer}.last"
    path.mkdir(parents=True)
    rows = []
    for seed in (1, 2):
        rows.extend([
            {"seed": seed, "control": "observed", "total_bits": bits,
             "marginal_bits": 100, "macro_f1": f1},
            {"seed": seed, "control": "shuffled_labels", "total_bits": 100,
             "marginal_bits": 100},
            {"seed": seed, "control": "shuffled_representations", "total_bits": 95,
             "marginal_bits": 100},
        ])
    pd.DataFrame(rows).to_csv(path / "mdl.mlp.tsv", sep="\t", index=False)


def test_quality_gate_requires_class_count_and_final_f1(tmp_path):
    (tmp_path / "label_gate.json").write_text(json.dumps({"predicates": {
        "knows": {"eligible": True, "negative": 1200, "positive": 1100},
        "true": {"eligible": False, "negative": 2000, "positive": 900},
    }}))
    for predicate in ("knows", "true"):
        _mdl(tmp_path, predicate, "p75", 40, 0.90)
        _mdl(tmp_path, predicate, "final", 50, 0.80)
    result = evaluate_quality_gate(tmp_path, ["p75", "final"], "last", 0.75)["predicates"]
    assert result["knows"]["comparative_claim_eligible"] is True
    assert result["knows"]["mdl_best_layer"] == "p75"
    assert result["true"]["comparative_claim_eligible"] is False


def test_quality_gate_uses_final_f1_not_best_layer_f1(tmp_path):
    (tmp_path / "label_gate.json").write_text(json.dumps({"predicates": {
        "knows": {"eligible": True, "negative": 1200, "positive": 1100}}}))
    _mdl(tmp_path, "knows", "p75", 40, 0.95)
    _mdl(tmp_path, "knows", "final", 50, 0.70)
    result = evaluate_quality_gate(tmp_path, ["p75", "final"], "last", 0.75)["predicates"]["knows"]
    assert result["mdl_best_layer"] == "p75"
    assert result["comparative_claim_eligible"] is False
