import json

import pytest

from src.probe_experiments import accept_amendment
from src.probe_utils import config_digest, config_diff, load_config


BEFORE = {"seeds": [1, 2], "llc": {"learning_rate": 0.0003, "localization": 1.0}}
AFTER = {"seeds": [1, 2], "llc": {"learning_rate": 0.0003, "localization": 1.0,
                                  "calibration_candidates": [{"localization": 100000.0}]}}


def _recorded():
    return {"config_digest": config_digest(BEFORE), "configuration": BEFORE}


def _declare(root, **overrides):
    entry = {"previous_digest": config_digest(BEFORE), "new_digest": config_digest(AFTER),
             "changed_settings": ["llc.calibration_candidates"],
             "reason": "the single default candidate cannot satisfy the basin criterion"}
    (root / "config_amendments.json").write_text(json.dumps([{**entry, **overrides}]))


def test_config_diff_names_the_changed_setting():
    assert config_diff(BEFORE, AFTER) == ["llc.calibration_candidates"]


def test_declared_amendment_is_accepted(tmp_path):
    _declare(tmp_path)
    accepted = accept_amendment(tmp_path, _recorded(), AFTER, config_digest(AFTER))
    assert accepted["changed_settings"] == ["llc.calibration_candidates"]


def test_undeclared_change_is_rejected(tmp_path):
    with pytest.raises(ValueError):
        accept_amendment(tmp_path, _recorded(), AFTER, config_digest(AFTER))


@pytest.mark.parametrize("overrides", [
    {"previous_digest": "0" * 64},
    {"new_digest": "0" * 64},
    {"changed_settings": ["seeds"]},
    {"changed_settings": []},
    {"reason": "  "},
])
def test_amendment_must_chain_name_the_change_and_give_a_reason(tmp_path, overrides):
    _declare(tmp_path, **overrides)
    with pytest.raises(ValueError):
        accept_amendment(tmp_path, _recorded(), AFTER, config_digest(AFTER))


def test_declared_llc_candidates_can_satisfy_the_basin_criterion():
    """Every candidate must be able to hold a 1,311,233-parameter chain inside 0.25*||w||."""
    candidates = load_config("configs/knows_mdl.yaml")["llc"]["calibration_candidates"]
    assert candidates, "the calibration grid must be declared, not left to the single default"
    parameters, basin_radius = 1_311_233, 0.25 * 14.209465026855469
    assert min(float(c["localization"]) for c in candidates) >= parameters / basin_radius ** 2
