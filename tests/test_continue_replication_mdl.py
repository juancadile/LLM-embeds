from pathlib import Path


def test_replication_mdl_queue_filters_on_class_gate_and_avoids_primary_knows_repeat():
    text = Path("scripts/continue_replication_mdl.sh").read_text()
    assert "while [[ ! -f replication_data.exit ]]" in text
    assert "$(<replication_data.exit) != 0" in text
    assert 'result["eligible"]' in text
    assert "run_mdl 'qwen-qwen3-14b' 'primary_other_predicates' 'knows'" in text
    assert "run_mdl 'qwen-qwen3-1.7b'" in text
    assert "run_mdl 'meta-llama-llama-3.1-8b-instruct'" in text
    assert "--pool last" in text
