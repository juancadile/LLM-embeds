from pathlib import Path


def test_mdl_continuation_gates_before_all_layer_sweep():
    text = Path("scripts/continue_after_primary_mdl.sh").read_text()
    assert "while [[ ! -f mdl_primary.exit ]]" in text
    assert "--minimum-macro-f1 0.75 --minimum-shuffled-improvement 0.10" in text
    assert text.index("-m src.probe_mdl_gate") < text.index("--stage mdl")
    assert "seq -w 1 39" in text
    assert "trap 'followup_status=$?" in text
