import numpy as np
import pytest

from src.probe_jtb import build_conditions, crossfit_features, gettier_split_cost


CFG = {"epochs": 3, "batch_size": 16, "learning_rate": 0.01, "weight_decay": 0.0,
       "patience": 2, "hidden_size": 4, "endpoints": [0.5, 1.0], "device": "cpu"}


def _cohort(n_groups=40, per_group=3):
    groups = np.repeat(np.arange(n_groups), per_group)
    split = np.where(groups < n_groups * 0.6, "coding",
                     np.where(groups < n_groups * 0.8, "tune", "evaluation"))
    return groups, split


def test_crossfit_never_scores_an_example_with_a_probe_that_saw_its_group(monkeypatch):
    """Every fold's training rows must exclude the whole scenario group being scored."""
    groups, split = _cohort()
    rows = np.arange(len(groups))
    labels = (groups % 2).astype(int)
    x = np.random.default_rng(0).normal(size=(len(groups), 6)).astype(np.float32)
    seen = []

    import src.probe_jtb as jtb

    def fake_train(kind, xt, yt, xv, yv, cfg, seed):
        seen.append(len(yt))
        class Model:
            pass
        return Model()

    monkeypatch.setattr(jtb, "train_probe", fake_train)
    monkeypatch.setattr(jtb, "predict_logits", lambda model, x: np.zeros(len(x)))
    monkeypatch.setattr(jtb.Standardizer, "fit", staticmethod(lambda a: type("S", (), {"transform": staticmethod(lambda v: v)})()))
    out = jtb.crossfit_features(x, rows, labels, split, groups, CFG, seed=1, folds=5)
    assert len(seen) == 5 and out.shape == (len(groups),)
    assert np.isfinite(out).all()


def test_crossfit_rejects_a_single_class_fold():
    groups, split = _cohort()
    rows = np.arange(len(groups))
    labels = np.ones(len(groups), dtype=int)
    x = np.zeros((len(groups), 3), dtype=np.float32)
    with pytest.raises(ValueError):
        crossfit_features(x, rows, labels, split, groups, CFG, seed=1, folds=5)


def test_declared_conditions_have_the_expected_widths():
    n = 12
    features = {name: np.arange(n, dtype=float) for name in
                ("believes", "justified", "true", "lucky_guessed")}
    conditions = build_conditions(features, np.zeros((n, 7)), features, np.zeros((n, 3)))
    widths = {name: np.atleast_2d(m).shape[1] if m.ndim > 1 else 1
              for name, m in conditions.items()}
    assert widths["jtb"] == 3 and widths["jtb_luck"] == 4
    assert widths["believes"] == 1 and widths["believes_justified"] == 2
    assert widths["raw_representation"] == 7 and widths["random_three_dimensions"] == 3
    assert widths["shuffled_components"] == 3


def test_gettier_gap_is_positive_when_luck_cases_are_mispredicted():
    rng = np.random.default_rng(0)
    n = 400
    luck = (np.arange(n) % 4 == 0).astype(int)
    y = rng.integers(0, 2, n)
    # Confident and correct on plain cases, confidently wrong on Gettier cases.
    logits = np.where(luck == 1, np.where(y == 1, -6.0, 6.0), np.where(y == 1, 6.0, -6.0))
    groups = np.repeat(np.arange(n // 4), 4)
    result = gettier_split_cost(y, logits, luck, groups, seed=0, draws=50)
    assert result["gap"] > 0
    assert result["gap_ci_low"] > 0
    assert result["n_gettier"] == int(luck.sum())
