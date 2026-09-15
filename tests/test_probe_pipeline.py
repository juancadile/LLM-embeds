from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from src.probe_compress import (load_pruned, load_quantized, load_svd_state,
                                save_pruned, save_quantized)
from src.probe_data import (assert_no_representation_leakage, assign_splits,
                            combine_counterbalanced, generate_scenarios,
                            validate_grouped_splits)
from src.probe_id import train_projected
from src.probe_llc import logistic_regular_expected_llc, validate_regular_logistic
from src.probe_mdl import marginal_bits, online_code
from src.probe_models import binary_metrics


FAST_PROBE = {"hidden_size": 8, "epochs": 20, "patience": 4, "batch_size": 64,
              "learning_rate": .02, "weight_decay": 0,
              "endpoints": [.1, .2, .4, .7, 1.]}


class DataTests(unittest.TestCase):
    def test_batched_continuation_scores_match_single_sequence_scores(self):
        from types import SimpleNamespace
        import torch
        from src.probe_hf import _continuation_logprob, _continuation_logprobs

        class Tokenizer:
            pad_token_id = 0
            def __call__(self, text, add_special_tokens=True):
                return {"input_ids": ([1] if add_special_tokens else []) +
                        [2 + ord(char) % 61 for char in text]}

        class Model:
            def __call__(self, input_ids, attention_mask=None, position_ids=None):
                vocab = 64
                values = torch.arange(vocab, device=input_ids.device).view(1, 1, -1)
                centers = ((input_ids + position_ids) % vocab).unsqueeze(-1)
                return SimpleNamespace(logits=-((values - centers).float() ** 2) / 20)

        requests = [("short:", " Yes"), ("a substantially longer prompt:", " No")]
        batched = _continuation_logprobs(Model(), Tokenizer(), requests, "cpu")
        singles = [_continuation_logprob(Model(), Tokenizer(), *request, "cpu")
                   for request in requests]
        np.testing.assert_allclose(batched, singles, atol=1e-6)

    def test_full_study_excludes_development_minimal_pair_families(self):
        development = generate_scenarios(128, 3, 2026)
        full = generate_scenarios(2048, 3, 2026, development_scenarios=128)
        self.assertFalse(set(development.minimal_pair_group) & set(full.minimal_pair_group))
        self.assertEqual(full.scenario_id.nunique(), 2048)
    def test_scenario_digest_detects_reordering_and_text_changes(self):
        from src.probe_utils import scenario_digest
        frame = generate_scenarios(8, 3, 2026)
        original = scenario_digest(frame)
        self.assertNotEqual(original, scenario_digest(frame.iloc[::-1]))
        changed = frame.copy()
        changed.loc[0, "scenario_text"] += " Additional evidence appeared."
        self.assertNotEqual(original, scenario_digest(changed))

    def test_all_paraphrases_state_proposition_and_gettier_is_consistent(self):
        frame = generate_scenarios(2048, 3, 2026)
        self.assertFalse(frame.scenario_text.duplicated().any())
        for row in frame.itertuples():
            self.assertIn(row.claim, row.scenario_text)
            if row.gettier:
                self.assertTrue(row.truth and row.belief and not row.guessing)

    def test_generated_minimal_pairs_differ_only_in_defeater(self):
        frame = generate_scenarios(256, 1, 2026)
        families = frame.groupby("minimal_pair_group", sort=False)
        self.assertTrue(all(len(group) == 2 for _, group in families))
        for _, group in families:
            self.assertEqual(set(group.defeater), {False, True})
            invariant = ["name", "claim", "truth", "belief", "evidence_source",
                         "reliable", "gettier", "guessing"]
            self.assertEqual(group[invariant].drop_duplicates().shape[0], 1)

    def test_canonical_support_allocation_is_prospective_and_exact(self):
        frame = generate_scenarios(128, 1, 2026, canonical_support_fraction=.24)
        support = (frame.truth & frame.belief & frame.reliable &
                   ~frame.gettier & ~frame.guessing)
        self.assertEqual(int(support.sum()), 30)
        self.assertTrue((frame.assign(support=support).groupby("minimal_pair_group")["support"]
                         .nunique() == 1).all())

    def test_minimal_pair_components_stay_together(self):
        frame = pd.DataFrame({"scenario_id": ["a", "b", "b", "c", "d"],
                              "minimal_pair_group": ["x", "x", "y", "y", None]})
        result = assign_splits(frame, {"tune": .1, "coding": .7, "evaluation": .2}, 11)
        self.assertEqual(result.iloc[:4].split_group.nunique(), 1)
        self.assertNotEqual(result.iloc[0].split_group, result.iloc[4].split_group)

    def test_scenario_pass_has_no_label_leakage_and_splits_are_grouped(self):
        a = generate_scenarios(64, 3, 7)
        b = generate_scenarios(64, 3, 7)
        pd.testing.assert_frame_equal(a, b)
        assert_no_representation_leakage(a.scenario_text)
        split = assign_splits(a, {"tune": .1, "coding": .7, "evaluation": .2}, 9)
        validate_grouped_splits(split)
        self.assertTrue((split.groupby("scenario_id").split.nunique() == 1).all())

    def test_counterbalancing_is_semantic(self):
        decision = combine_counterbalanced([3, 0], [0, 3], .8)
        self.assertTrue(decision.stable)
        self.assertTrue(decision.label)
        unstable = combine_counterbalanced([3, 0], [3, 0], .8)
        self.assertFalse(unstable.stable)

    def test_semantic_response_candidates_follow_display_order(self):
        from src.probe_hf import response_candidates
        cfg = {"labels": {"prompt_variant": "semantic_tokens"}}
        self.assertEqual(response_candidates(cfg, False), [" Yes", " No"])
        self.assertEqual(response_candidates(cfg, True), [" No", " Yes"])


class MdlTests(unittest.TestCase):
    def test_nll_is_not_clipped_for_confident_wrong_predictions(self):
        result = binary_metrics(np.array([0, 1]), np.array([100., -100.]))
        self.assertAlmostEqual(result["nll_nats"], 100.)
    def test_null_permutation_preserves_groups_and_splits(self):
        from src.probe_mdl import group_permutation
        groups = np.repeat(np.arange(12), 3)
        splits = np.repeat(["coding", "tune"], 18)
        permutation = group_permutation(groups, splits, 11)
        self.assertEqual(sorted(permutation.tolist()), list(range(36)))
        np.testing.assert_array_equal(splits[permutation], splits)
        for group in np.unique(groups):
            self.assertEqual(len(np.unique(groups[permutation[groups == group]])), 1)
    def test_blocks_do_not_divide_groups_and_prior_ignores_future_labels(self):
        from unittest.mock import patch
        rng = np.random.default_rng(17)
        groups = np.repeat(np.arange(5), [3, 6, 2, 4, 3])
        x = rng.normal(size=(len(groups), 2)).astype(np.float32)
        y = rng.integers(0, 2, len(x))
        tune_y = np.array([0, 1, 1, 1])
        tune_x = x[:4]
        with patch("src.probe_mdl.train_probe", return_value=object()), patch(
                "src.probe_mdl.predict_logits", side_effect=lambda model, data: np.zeros(len(data))):
            code, _, _ = online_code(x, y, tune_x, tune_y, groups, "linear", FAST_PROBE, 5)
            altered, _, _ = online_code(x, 1-y, tune_x, tune_y, groups, "linear", FAST_PROBE, 5)
        ordered = groups[code["row_order"]]
        for block in code["blocks"]:
            left, right = block["test_start"], block["test_end"]
            self.assertFalse(set(ordered[:left]) & set(ordered[left:right]))
            self.assertFalse(set(ordered[:right]) & set(ordered[right:]))
        self.assertEqual(code["blocks"][0]["prior"], altered["blocks"][0]["prior"])

    def test_online_code_compresses_linear_rule_but_not_random_labels(self):
        rng = np.random.default_rng(4)
        x = rng.normal(size=(500, 5)).astype(np.float32)
        groups = np.arange(len(x))
        tune_x = rng.normal(size=(120, 5)).astype(np.float32)
        linear_y = (x[:, 0] > 0).astype(int)
        tune_y = (tune_x[:, 0] > 0).astype(int)
        linear, _, _ = online_code(x, linear_y, tune_x, tune_y, groups, "linear", FAST_PROBE, 5)
        random_y = rng.integers(0, 2, len(x)); random_tune = rng.integers(0, 2, len(tune_x))
        random, _, _ = online_code(x, random_y, tune_x, random_tune, groups, "linear", FAST_PROBE, 5)
        self.assertGreater(linear["compression"], .20)
        self.assertLess(abs(random["total_bits"] - marginal_bits(random_y)) / marginal_bits(random_y), .20)


class IntrinsicDimensionTests(unittest.TestCase):
    def test_projection_has_requested_rank_and_orthonormal_columns(self):
        from src.probe_id import projection_indices
        for dimension in (1, 3, 17):
            buckets, signs = projection_indices(17, dimension, 9)
            matrix = np.zeros((17, dimension))
            matrix[np.arange(17), buckets.numpy()] = signs.numpy()
            np.testing.assert_allclose(matrix.T @ matrix, np.eye(dimension), atol=1e-6)
    def test_projected_training_learns_low_dimensional_toy(self):
        rng = np.random.default_rng(8)
        x = rng.normal(size=(240, 2)).astype(np.float32)
        y = (x[:, 0] + .2 * x[:, 1] > 0).astype(int)
        predict, total = train_projected(x[:180], y[:180], x[180:], y[180:], 2, 4, 16, FAST_PROBE, 2)
        self.assertGreater(total, 16)
        self.assertGreater(binary_metrics(y[180:], predict(x[180:]))["accuracy"], .75)


class CompressionTests(unittest.TestCase):
    def test_loading_state_vector_does_not_alias_source_array(self):
        import torch
        from src.probe_models import load_state_vector, make_probe, state_vector
        model = make_probe("mlp", 3, 4)
        source = state_vector(model).copy()
        expected = source.copy()
        load_state_vector(model, source)
        replacement = {key: torch.zeros_like(value) for key, value in model.state_dict().items()}
        model.load_state_dict(replacement)
        np.testing.assert_array_equal(source, expected)
        self.assertGreater(np.count_nonzero(source), 0)

    def test_test_labels_cannot_change_compression_selection(self):
        from src.probe_compress import run_compression
        from src.probe_models import make_probe, save_probe, Standardizer
        rng = np.random.default_rng(22)
        x = rng.normal(size=(30, 2)).astype(np.float32)
        y = (x[:, 0] > 0).astype(int)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            model = make_probe("mlp", 2, 4)
            path = root / "model.pt"
            save_probe(path, model, Standardizer.fit(x),
                       {"kind": "mlp", "input_size": 2, "hidden_size": 4})
            cfg = {"quantization_bits": [8, 2], "pruning": [.5],
                   "max_delta_nll": .05, "max_accuracy_loss": .01}
            a = run_compression(path, x, y, cfg, root / "a", x_test=x, y_test=y)
            b = run_compression(path, x, y, cfg, root / "b", x_test=x, y_test=1-y)
            self.assertEqual(a.loc[a.selected, "artifact"].tolist(), b.loc[b.selected, "artifact"].tolist())
            self.assertEqual(a.selected.sum(), 1)
            self.assertEqual(a.test_accuracy.notna().sum(), 1)

    def test_formats_roundtrip_and_byte_counts_are_real(self):
        vector = np.linspace(-2, 2, 1001, dtype=np.float32)
        with tempfile.TemporaryDirectory() as d:
            d = Path(d)
            q = d / "q.npz"; save_quantized(q, vector, 6)
            restored = load_quantized(q)
            self.assertEqual(restored.shape, vector.shape)
            self.assertLess(np.max(np.abs(restored - vector)), .04)
            self.assertEqual(q.stat().st_size, len(q.read_bytes()))
            p = d / "p.npz"; save_pruned(p, vector, .9)
            sparse = load_pruned(p)
            self.assertEqual(np.count_nonzero(sparse), 100)
            self.assertEqual(p.stat().st_size, len(p.read_bytes()))

    def test_svd_artifacts_reload_complete_probe_and_report_real_bytes(self):
        from src.probe_compress import run_compression
        from src.probe_models import make_probe, save_probe, Standardizer
        rng = np.random.default_rng(23)
        x = rng.normal(size=(30, 4)).astype(np.float32)
        y = (x[:, 0] > 0).astype(int)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = root / "probe.pt"
            save_probe(path, make_probe("mlp", 4, 4), Standardizer.fit(x),
                       {"kind": "mlp", "input_size": 4, "hidden_size": 4})
            result = run_compression(path, x, y, {
                "quantization_bits": [8], "pruning": [.5],
                "max_delta_nll": 10, "max_accuracy_loss": 1,
            }, root / "compressed", x_test=x, y_test=y)
            svd = result[result.method == "svd"]
            self.assertGreater(len(svd), 0)
            for row in svd.itertuples():
                artifact = root / "compressed" / row.artifact
                loaded = load_svd_state(artifact)
                self.assertEqual(set(loaded["state_dict"]),
                                 {"0.weight", "0.bias", "2.weight", "2.bias"})
                self.assertEqual(row.bytes, len(artifact.read_bytes()))

    def test_regular_logistic_reference(self):
        self.assertEqual(logistic_regular_expected_llc(12), 6)

    def test_psgld_matches_regular_logistic_reference(self):
        try:
            from src.probe_llc import devinterp_provenance
            devinterp_provenance()
        except RuntimeError as exc:
            self.skipTest(str(exc))
        result = validate_regular_logistic()
        self.assertTrue(result["passed"], result)


class ReportTests(unittest.TestCase):
    def test_high_f1_cannot_override_inadequate_class_counts(self):
        from src.probe_report import build_report
        from src.probe_utils import write_table
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            write_table(pd.DataFrame({"predicate": ["knows"] * 2,
                "stable": [True, True], "label": [0, 1],
                "example_id": ["a", "b"]}), root / "model" / "labels.parquet")
            cell = root / "model" / "cells" / "knows" / "final.last"
            write_table(pd.DataFrame({"control": ["observed"], "macro_f1": [1.0],
                "bits_per_label": [.2], "compression": [.8]}), cell / "mdl.mlp.parquet")
            cfg = {"models": {"primary": "model"}, "thresholds": {
                "minimum_stable_per_class": 1000, "minimum_macro_f1": .75}}
            report = build_report(cfg, root)
            row = next(line for line in report.splitlines() if "model/cells/knows/final.last" in line)
            self.assertTrue(row.endswith("| no |"), row)
            self.assertIn("Confirmatory correlations are unavailable", report)


class LlcDiagnosticsTests(unittest.TestCase):
    def test_llc_batch_size_covers_all_rows_under_cycle_mode(self):
        from src.probe_llc import divisor_batch_size
        self.assertEqual(divisor_batch_size(300, 64), 60)
        self.assertEqual(divisor_batch_size(4096, 128), 128)
        self.assertEqual(300 % divisor_batch_size(300, 64), 0)

    def test_sensitivity_grid_is_axial_not_cartesian(self):
        from src.probe_llc import sensitivity_settings
        self.assertEqual(len(sensitivity_settings([.5, 1, 2])), 5)
        self.assertNotIn((.5, .5), sensitivity_settings([.5, 1, 2]))

    def test_calibration_is_frozen_and_rejects_changed_inputs(self):
        from unittest.mock import patch
        from src.probe_calibration import calibrate
        from src.probe_models import make_probe, save_probe, Standardizer
        x = np.ones((8, 2), dtype=np.float32)
        y = np.array([0, 1] * 4)
        cfg = {"learning_rate": .001, "localization": 1}
        accepted = pd.DataFrame({"estimate_rejected": [False], "rejection_reason": [None]})
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); path = root / "probe.pt"
            save_probe(path, make_probe("mlp", 2, 4), Standardizer.fit(x),
                {"kind": "mlp", "input_size": 2, "hidden_size": 4, "predicate": "knows"})
            with patch("src.probe_calibration.run_llc", return_value=accepted) as run:
                a = calibrate(path, x, y, cfg, [1, 2, 3, 4], root / "calibration")
                b = calibrate(path, x, y, cfg, [1, 2, 3, 4], root / "calibration")
                self.assertEqual(a, b)
                self.assertEqual(run.call_count, 1)
                with self.assertRaisesRegex(ValueError, "inputs changed"):
                    calibrate(path, x, 1-y, cfg, [1, 2, 3, 4], root / "calibration")

    def test_requires_both_neighbors_in_both_directions(self):
        from src.probe_llc import sensitivity_rejection
        frame = pd.DataFrame([{"lr_multiplier": lr, "localization_multiplier": gamma,
            "chain": chain, "accepted": True, "llc": 2.0}
            for lr, gamma in ((1, 1), (.5, 1), (2, 1), (1, .5), (1, 2)) for chain in range(4)])
        self.assertIsNone(sensitivity_rejection(frame, 4))
        frame.loc[frame.lr_multiplier == 2, "llc"] = 2.3
        self.assertEqual(sensitivity_rejection(frame, 4), "sensitivity_exceeds_10_percent")
        self.assertEqual(sensitivity_rejection(frame.iloc[:-1], 4), "incomplete_sensitivity_grid")

    def test_rejects_nonmixing_and_nonfinite_chains(self):
        from src.probe_llc import trace_diagnostics
        rng = np.random.default_rng(3)
        independent = rng.normal(size=(4, 1000))
        self.assertTrue(trace_diagnostics(independent)["passed"])
        independent[0] += 10
        self.assertFalse(trace_diagnostics(independent)["passed"])
        independent[0, 0] = np.nan
        self.assertFalse(trace_diagnostics(independent)["passed"])


class PairedBootstrapTests(unittest.TestCase):
    def test_cells_share_resamples_and_failed_cells_are_not_dropped(self):
        from src.probe_bootstrap import paired_refit_bootstrap, correlation_intervals
        frame = assign_splits(generate_scenarios(32, 3, 4),
                              {"tune": .1, "coding": .7, "evaluation": .2}, 8)
        calls = []
        def evaluate(cell, indices, seed, groups):
            calls.append(indices)
            assert len(groups) == len(indices)
            value = int(cell) + 1
            return {"normalized_mdl": value, "d90": value,
                    "compressed_bytes": value, "llc": value}
        cells = list(map(str, range(15)))
        result = paired_refit_bootstrap(frame, cells, evaluate, draws=2, seed=2)
        for draw in range(2):
            for index in range(15):
                np.testing.assert_array_equal(calls[draw*15], calls[draw*15+index])
        result.loc[0, "d90"] = np.nan
        self.assertEqual(correlation_intervals(result, cells)["reason"], "invalid_or_censored_estimator")


if __name__ == "__main__": unittest.main()
