from pathlib import Path

import pandas as pd

from src.probe_experiments import llc_gate
from src.probe_utils import write_table


def _cell(tmp_path: Path, *, tune_f1: float, eval_f1: float) -> Path:
    write_table(pd.DataFrame({
        "control": ["observed", "shuffled_labels"],
        "total_bits": [80.0, 100.0],
        "tune_macro_f1": [tune_f1, 0.5],
        "macro_f1": [eval_f1, 0.5],
    }), tmp_path / "mdl.mlp.parquet")
    write_table(pd.DataFrame({"projection_seed": [0, 1], "dimension": [8, 16],
                              "is_d90": [True, True]}),
                tmp_path / "intrinsic_dimension.parquet")
    write_table(pd.DataFrame({"seed": [11, 12], "bytes": [16, 18],
                              "selected": [True, True], "acceptable": [True, True]}),
                tmp_path / "compression.parquet")
    return tmp_path


def test_llc_gate_uses_untouched_evaluation_f1(tmp_path):
    cfg = {"thresholds": {"minimum_macro_f1": 0.75, "mdl_shuffled_improvement": 0.10},
           "intrinsic_dimension": {"projection_seeds": [0, 1]}, "seeds": [11, 12]}
    passed, reason = llc_gate(_cell(tmp_path, tune_f1=0.99, eval_f1=0.70), cfg)
    assert not passed
    assert reason == "macro_f1_below_threshold"


def test_llc_gate_accepts_completed_cell(tmp_path):
    cfg = {"thresholds": {"minimum_macro_f1": 0.75, "mdl_shuffled_improvement": 0.10},
           "intrinsic_dimension": {"projection_seeds": [0, 1]}, "seeds": [11, 12]}
    passed, reason = llc_gate(_cell(tmp_path, tune_f1=0.70, eval_f1=0.90), cfg)
    assert passed
    assert reason == "passed"


def test_llc_gate_rejects_censored_id_and_missing_compression_seed(tmp_path):
    cfg = {"thresholds": {"minimum_macro_f1": 0.75, "mdl_shuffled_improvement": 0.10},
           "intrinsic_dimension": {"projection_seeds": [0, 1]}, "seeds": [11, 12]}
    cell = _cell(tmp_path, tune_f1=0.9, eval_f1=0.9)
    dimension = pd.read_csv(cell / "intrinsic_dimension.tsv", sep="\t")
    dimension.loc[dimension.projection_seed == 1, "is_d90"] = False
    write_table(dimension, cell / "intrinsic_dimension.parquet")
    assert llc_gate(cell, cfg)[1] == "intrinsic_dimension_not_reached"
    dimension["is_d90"] = True
    write_table(dimension, cell / "intrinsic_dimension.parquet")
    compression = pd.read_csv(cell / "compression.tsv", sep="\t")
    compression.loc[compression.seed == 12, "selected"] = False
    write_table(compression, cell / "compression.parquet")
    assert llc_gate(cell, cfg)[1] == "compression_criterion_not_met"
