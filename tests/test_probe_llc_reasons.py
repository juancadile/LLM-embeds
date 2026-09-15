import numpy as np

import src.probe_llc as probe_llc


class _Scaler:
    def transform(self, x):
        return x


def _diverged_chain(*args, **kwargs):
    return {"accepted": False, "reason": "sampler_error:ValueError", "llc": float("nan"),
            "loss_trace": [], "distance_trace": [], "diagnostic_draws": [],
            "diagnostic_losses": [], "devinterp": {}}


def test_sampler_failure_reason_survives_the_incomplete_trace_check(tmp_path, monkeypatch):
    monkeypatch.setattr(probe_llc, "devinterp_provenance", lambda: {})
    monkeypatch.setattr(probe_llc, "load_probe", lambda path: (None, _Scaler(), {}))
    monkeypatch.setattr(probe_llc, "psgld_chain", _diverged_chain)
    cfg = {"chains": 2, "draws": 100, "burn_in": 10, "thinning": 10,
           "learning_rate": 1e-4, "localization": 1e5,
           "sensitivity_multipliers": [0.5, 1.0, 2.0],
           "max_basin_distance": 3.5, "max_loss_increase": 0.05}
    frame = probe_llc.run_llc(tmp_path / "probe.pt", np.zeros((4, 2)), np.zeros(4), cfg,
                              [1, 2], tmp_path / "out")
    assert set(frame.reason) == {"sampler_error:ValueError"}
    assert frame.rejection_reason.iloc[0] == "invalid_sensitivity_chains"
