from pathlib import Path


def test_llc_queue_waits_for_environment_and_full_prerequisite_grid():
    text = Path("scripts/continue_llc.sh").read_text()
    assert "wait_for_job llc_environment.exit llc_environment.pid" in text
    assert "wait_for_job artifact_audit.exit artifact_audit.pid" in text
    assert "comparative_claim_eligible" in text
    assert "PYTHONPATH=analysis_v8" in text
    assert "--stage llc" in text
    assert "llc.exit" in text
