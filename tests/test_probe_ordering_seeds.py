import numpy as np
import pandas as pd

from src.probe_ordering import ordering_summary


def test_summary_averages_over_probe_seeds_within_a_draw_before_ranking():
    rows = []
    for draw in range(3):
        for seed, offset in ((1, -0.05), (2, +0.05)):
            rows.append({"draw": draw, "predicate": "a", "seed": seed, "normalized_mdl": 0.10 + offset, "bits_per_label": 0})
            rows.append({"draw": draw, "predicate": "b", "seed": seed, "normalized_mdl": 0.12 + offset, "bits_per_label": 0})
    summary = ordering_summary(pd.DataFrame(rows), "normalized_mdl")
    assert summary["complete_draws"] == 3
    # Per-seed values straddle each other; the per-draw means do not.
    assert summary["per_predicate"]["a"]["median"] == 0.10
    assert summary["per_predicate"]["b"]["median"] == 0.12
    assert summary["probability_cheapest"]["a"] == 1.0


def test_summary_still_accepts_a_single_seed_table_without_a_seed_column():
    frame = pd.DataFrame({"draw": [0, 0, 1, 1], "predicate": ["a", "b", "a", "b"],
                          "normalized_mdl": [0.1, 0.2, 0.1, 0.2], "bits_per_label": 0})
    assert ordering_summary(frame, "normalized_mdl")["complete_draws"] == 2
