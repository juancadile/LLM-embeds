from pathlib import Path


def test_bound_wrapper_is_versioned_and_writes_a_terminal_marker():
    text = Path("scripts/night_bound.sh").read_text()
    assert "PYTHONPATH=analysis_v11" in text
    assert "src.probe_bound" in text
    assert "bound.exit" in text
