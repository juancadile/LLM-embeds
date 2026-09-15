from pathlib import Path


def test_artifact_audit_waits_for_grid_and_covers_all_eligible_cells():
    text = Path("scripts/continue_artifact_audit.sh").read_text()
    assert "while [[ ! -f id_compression_grid.exit ]]" in text
    assert "comparative_claim_eligible" in text
    assert "layers+=(block29)" in text
    assert "PYTHONPATH=analysis_v6" in text
    assert "src.probe_artifact_audit" in text
    assert "artifact_audit.exit" in text
