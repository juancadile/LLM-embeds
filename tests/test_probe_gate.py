import json

import pandas as pd

from src.probe_gate import evaluate_label_gate, main


def test_label_gate_counts_classes_and_splits():
    labels = pd.DataFrame([
        {"predicate": "knows", "label": True, "stable": True, "split": "coding"},
        {"predicate": "knows", "label": "true", "stable": "true", "split": "evaluation"},
        {"predicate": "knows", "label": False, "stable": True, "split": "coding"},
        {"predicate": "knows", "label": True, "stable": False, "split": "tune"},
        {"predicate": "true", "label": False, "stable": True, "split": "tune"},
    ])
    result = evaluate_label_gate(labels, minimum_stable_per_class=1)["predicates"]
    assert result["knows"]["eligible"] is True
    assert result["knows"]["positive"] == 2
    assert result["knows"]["negative"] == 1
    assert result["knows"]["unstable"] == 1
    assert result["knows"]["by_split"]["coding"] == {
        "stable": 2, "negative": 1, "positive": 1}
    assert result["true"]["eligible"] is False


def test_label_gate_cli_writes_failed_gate(tmp_path, monkeypatch):
    labels = pd.DataFrame([
        {"predicate": "knows", "label": True, "stable": True, "split": "coding"},
        {"predicate": "knows", "label": False, "stable": True, "split": "coding"},
    ])
    path = tmp_path / "labels.tsv"
    labels.to_csv(path, sep="\t", index=False)
    output = tmp_path / "gate.json"
    monkeypatch.setattr("sys.argv", ["probe_gate", "--labels", str(path),
        "--output", str(output), "--minimum", "2", "--require", "knows"])
    try:
        main()
    except SystemExit as exc:
        assert exc.code == 2
    else:
        raise AssertionError("failed gate should return a nonzero status")
    result = json.loads(output.read_text())
    assert result["passed"] is False
    assert result["failed_required_predicates"] == ["knows"]
