import json

from src.probe_report import _calibration_status


CFG = {"models": {"primary": "Qwen/Qwen3-14B"}}


def _manifest(tmp_path, payload):
    folder = tmp_path / "qwen-qwen3-14b" / "llc_calibration"
    folder.mkdir(parents=True)
    (folder / "calibration.json").write_text(json.dumps(payload))
    return tmp_path


def test_rejected_calibration_is_stated_as_a_result(tmp_path):
    root = _manifest(tmp_path, {"status": "rejected", "attempts": [
        {"reason": "sensitivity_exceeds_10_percent"},
        {"reason": "sensitivity_exceeds_10_percent"},
        {"reason": "invalid_sensitivity_chains"}]})
    text = " ".join(_calibration_status(root, CFG))
    assert "No LLC estimate is reported" in text
    assert "sensitivity_exceeds_10_percent (2)" in text
    assert "non-identifiable" in text


def test_accepted_calibration_reports_the_frozen_setting(tmp_path):
    root = _manifest(tmp_path, {"status": "accepted", "attempts": [{"reason": None}],
                                "settings": {"learning_rate": 1e-07, "localization": 300000.0}})
    text = " ".join(_calibration_status(root, CFG))
    assert "accepted learning rate 1e-07 and localization 300000" in text


def test_absent_manifest_adds_nothing(tmp_path):
    assert _calibration_status(tmp_path, CFG) == []
