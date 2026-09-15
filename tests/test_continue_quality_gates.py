from pathlib import Path


def test_quality_gate_queue_waits_for_replication_mdl_and_checks_all_models():
    text = Path("scripts/continue_quality_gates.sh").read_text()
    assert "while [[ ! -f replication_mdl.exit ]]" in text
    assert "$(<replication_mdl.exit) != 0" in text
    assert text.count("run_gate '") == 3
    assert "--minimum-macro-f1 0.75" in text
    assert "--layer emb --layer p25 --layer p50 --layer p75 --layer final" in text
    assert "missing label gate" in text
