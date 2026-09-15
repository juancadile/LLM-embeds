from pathlib import Path


def test_ordering_wrapper_is_versioned_and_can_run_a_tagged_multi_seed_pass():
    text = Path("scripts/night_ordering.sh").read_text()
    assert "PYTHONPATH=analysis_v12" in text
    assert "src.probe_ordering" in text
    assert "--probe linear" in text
    assert 'ordering${TAG}.exit' in text
    assert '--seeds "$SEEDS"' in text
