from pathlib import Path


def test_hotfix_uses_versioned_source_and_writes_terminal_marker():
    text = Path("scripts/rerun_primary_compression_hotfix.sh").read_text()
    assert "PYTHONPATH=analysis_v5" in text
    assert "CUDA_VISIBLE_DEVICES=''" in text
    assert "--stage compress" in text
    assert "--layer block29" in text
    assert "compression_primary_hotfix.exit" in text
