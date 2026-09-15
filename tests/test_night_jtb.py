from pathlib import Path


def test_jtb_wrapper_is_versioned_and_writes_a_terminal_marker():
    text = Path("scripts/night_jtb.sh").read_text()
    assert "PYTHONPATH=analysis_v10" in text
    assert "src.probe_jtb" in text
    assert "jtb.exit" in text
