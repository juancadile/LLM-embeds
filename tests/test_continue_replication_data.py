from pathlib import Path


def test_replication_queue_waits_and_runs_both_models():
    text = Path("scripts/continue_replication_data.sh").read_text()
    assert "while [[ ! -f id_compress_primary.exit ]]" in text
    assert "$(<id_compress_primary.exit) != 0" in text
    assert "qwen-qwen3-1.7b" in text
    assert "meta-llama-llama-3.1-8b-instruct" in text
    assert text.count("run_model '") == 2
    assert text.index("--stage labels") < text.index("--stage extract")
    assert "--minimum 1000" in text
    assert "extraction suppressed" in text
