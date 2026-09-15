import numpy as np
import pytest
import torch

from src.probe_bound import (compress_folded, fold_probe, fold_standardizer, identity_standardizer,
                             ladder, payload_bytes, standardizer_is_absorbable)
from src.probe_models import Standardizer, load_probe, make_probe, predict_logits, save_probe


def _scaler(dim, rng):
    return Standardizer.fit(rng.normal(3.0, 2.0, size=(64, dim)).astype(np.float32))


def _save(path, kind, dim, hidden, scaler):
    torch.manual_seed(0)
    save_probe(path, make_probe(kind, dim, hidden), scaler,
               {"kind": kind, "input_size": dim, "hidden_size": hidden})


def test_folding_the_standardizer_preserves_every_logit_for_both_probe_kinds(tmp_path):
    rng = np.random.default_rng(0)
    x = rng.normal(3.0, 2.0, size=(32, 7)).astype(np.float32)
    for kind in ("linear", "mlp"):
        _save(tmp_path / f"probe.{kind}.pt", kind, 7, 5, _scaler(7, rng))
        result = standardizer_is_absorbable(tmp_path / f"probe.{kind}.pt", x)
        assert result["absorbable"], (kind, result)
        assert result["max_relative_logit_difference"] < 1e-5


def test_folded_checkpoint_scores_raw_activations_like_the_original_scored_standardized(tmp_path):
    rng = np.random.default_rng(1)
    x = rng.normal(3.0, 2.0, size=(40, 6)).astype(np.float32)
    _save(tmp_path / "probe.pt", "linear", 6, 2, _scaler(6, rng))
    model, scaler, _ = load_probe(tmp_path / "probe.pt")
    reference = predict_logits(model, scaler.transform(x))
    folded, identity, meta = load_probe(fold_probe(tmp_path / "probe.pt", tmp_path / "folded.pt"))
    assert np.all(identity.mean == 0) and np.all(identity.scale == 1)
    assert meta["standardizer"] == "folded_into_first_layer"
    np.testing.assert_allclose(predict_logits(folded, identity.transform(x)), reference, rtol=1e-4, atol=1e-4)


def test_payload_refuses_an_artifact_compressed_in_standardized_space(tmp_path):
    """Dropping a real standardizer leaves a description that cannot read raw activations."""
    artifact = tmp_path / "quantized.8bit.npz"
    np.savez(artifact, format="q", vector=np.zeros(16, dtype=np.int8),
             standardizer_mean=np.full(5, 3.0, dtype=np.float32),
             standardizer_scale=np.full(5, 2.0, dtype=np.float32))
    with pytest.raises(ValueError, match="standardized-input space"):
        payload_bytes(artifact, tmp_path / "out")


def test_payload_strips_only_an_identity_standardizer(tmp_path):
    artifact = tmp_path / "quantized.8bit.npz"
    np.savez(artifact, format="q", vector=np.zeros(16, dtype=np.int8),
             standardizer_mean=np.zeros(5120, dtype=np.float32),
             standardizer_scale=np.ones(5120, dtype=np.float32))
    assert payload_bytes(artifact, tmp_path / "out") < artifact.stat().st_size - 40000


def test_compress_folded_selects_on_raw_activations_and_reports_payloads(tmp_path):
    rng = np.random.default_rng(2)
    x = rng.normal(3.0, 2.0, size=(160, 6)).astype(np.float32)
    y = (x[:, 0] - 3.0 + 0.3 * rng.normal(size=160) > 0).astype(int)
    scaler = Standardizer.fit(x[:80])
    torch.manual_seed(0)
    model = make_probe("linear", 6, 2)
    with torch.no_grad():
        model.weight.zero_(); model.weight[0, 0] = 4.0; model.bias.zero_()
    save_probe(tmp_path / "probe.pt", model, scaler, {"kind": "linear", "input_size": 6, "hidden_size": 2})
    cfg = {"quantization_bits": [16, 8, 4], "pruning": [0.5], "max_delta_nll": 0.05, "max_accuracy_loss": 0.01}
    frame = compress_folded(tmp_path / "probe.pt", x[:80], y[:80], x[80:], y[80:], cfg, tmp_path / "out")
    assert "payload_bytes" in frame and (frame.payload_bytes < frame.bytes).all()
    assert frame.selected.sum() == 1
    assert (tmp_path / "out" / "probe.folded.pt").exists()


def test_ladder_carries_the_standardizer_only_on_subspace_rungs(tmp_path):
    save_probe(tmp_path / "probe.linear.seed1.full.pt", make_probe("linear", 10, 2),
               Standardizer.fit(np.arange(40, dtype=np.float32).reshape(4, 10)),
               {"kind": "linear", "input_size": 10, "hidden_size": 2})
    table = ladder(tmp_path, 10, standardizer_bytes=80, subspace={"linear": 6}, seeds=[1],
                   compressed={"linear": {"payload_bytes": 20.0, "evaluation_macro_f1": 0.9}})
    rungs = table.set_index("rung")
    assert rungs.loc["linear_float32", "standalone_bytes"] == 11 * 4
    assert rungs.loc["linear_compressed", "standalone_bytes"] == 20.0
    assert rungs.loc["linear_compressed", "bytes_if_standardizer_shared"] == 20.0
    assert rungs.loc["linear_subspace_d90", "standalone_bytes"] == 6 * 4 + 8 + 80
    assert rungs.loc["linear_subspace_d90", "bytes_if_standardizer_shared"] == 6 * 4 + 8
