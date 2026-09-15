import numpy as np
import pandas as pd
import pytest

from src.probe_ordering import ordering_summary


def _draws(values: dict[str, list[float]]) -> pd.DataFrame:
    rows = []
    for predicate, series in values.items():
        for draw, value in enumerate(series):
            rows.append({"draw": draw, "predicate": predicate,
                         "normalized_mdl": value, "bits_per_label": value})
    return pd.DataFrame(rows)


def test_separated_predicates_are_reported_as_separated():
    frame = _draws({"believes": [0.10, 0.11, 0.12], "lucky_guessed": [0.30, 0.31, 0.32]})
    summary = ordering_summary(frame, "normalized_mdl")
    assert summary["probability_cheapest"]["believes"] == 1.0
    assert summary["probability_dearest"]["lucky_guessed"] == 1.0
    pair = summary["pairs"]["believes_minus_lucky_guessed"]
    assert pair["probability_a_cheaper"] == 1.0 and pair["separated"]


def test_overlapping_predicates_are_not_separated():
    rng = np.random.default_rng(0)
    a = rng.normal(0.15, 0.02, 200)
    frame = _draws({"knows": list(a), "justified": list(a + rng.normal(0, 0.02, 200))})
    pair = ordering_summary(frame, "normalized_mdl")["pairs"]["justified_minus_knows"]
    assert not pair["separated"]
    assert 0.2 < pair["probability_a_cheaper"] < 0.8


def test_degenerate_draws_are_dropped_not_imputed():
    frame = _draws({"a": [0.1, np.nan, 0.3], "b": [0.2, 0.2, 0.2]})
    summary = ordering_summary(frame, "normalized_mdl")
    assert summary["complete_draws"] == 2


def test_empty_result_raises_rather_than_reporting_an_ordering():
    frame = _draws({"a": [np.nan], "b": [0.2]})
    with pytest.raises(ValueError):
        ordering_summary(frame, "normalized_mdl")
