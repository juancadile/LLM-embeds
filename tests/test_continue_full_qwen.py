from pathlib import Path


def test_continuation_waits_for_labels_and_gates_before_extracting():
    text = Path("scripts/continue_full_qwen.sh").read_text()
    assert "while [[ ! -f labels.exit ]]" in text
    assert "EXPERIMENT=${EXPERIMENT:?" in text
    assert "--minimum 1000 --require knows" in text
    assert text.index("-m src.probe_gate") < text.index("--stage extract")
    assert "trap 'continuation_status=$?" in text
