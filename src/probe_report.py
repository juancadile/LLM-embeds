"""Build a single auditable report from probing artifacts."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from .probe_utils import read_table


def _tables(root: Path, name: str) -> list[Path]:
    paths = list(root.rglob(name + ".parquet")) + [p for p in root.rglob(name + ".tsv") if not p.with_suffix(".parquet").exists()]
    return sorted(p for p in paths
                  if not {"joint_bootstrap", "compression_seeds"}.intersection(p.relative_to(root).parts))


def _calibration_status(root: Path, cfg: dict) -> list[str]:
    """State the terminal calibration outcome; a rejection is itself a result."""
    import json
    from .probe_experiments import slug
    manifest = root / slug(cfg["models"]["primary"]) / "llc_calibration" / "calibration.json"
    if not manifest.exists():
        return []
    frozen = json.loads(manifest.read_text())
    if frozen.get("status") == "accepted":
        settings = frozen["settings"]
        return [f"Calibration accepted learning rate {settings['learning_rate']:g} and localization "
                f"{settings['localization']:g} after {len(frozen['attempts'])} declared candidate(s).", ""]
    reasons: dict[str, int] = {}
    for attempt in frozen.get("attempts", []):
        reasons[attempt["reason"]] = reasons.get(attempt["reason"], 0) + 1
    counted = ", ".join(f"{reason} ({count})" for reason, count in sorted(reasons.items()))
    return [f"**No LLC estimate is reported.** All {len(frozen.get('attempts', []))} declared calibration "
            f"candidates were rejected: {counted}. Per-cell LLC was therefore never run, and the learning "
            "coefficient is reported as non-identifiable for this probe under the declared locality "
            "criteria rather than estimated at a setting that failed them.", ""]


def _rel_cell(path: Path, root: Path) -> str:
    return "/".join(path.relative_to(root).parts[:-1])


def _convergence(root: Path, mdl_paths, id_paths, compression_paths, llc_paths) -> dict | None:
    path = root / "joint_bootstrap" / "summary.json"
    if not path.exists(): return None
    result = json.loads(path.read_text())
    if (not result.get("accepted") or result.get("draw_count", 0) < 1000 or
        len(set(result.get("cell_ids", []))) != 15 or
        result.get("method") != "paired_scenario_refits_with_independent_chains"):
        return None
    return {"n_cells": 15, "convergent": result["convergent"],
        **{key: {"rho": result["intervals"][measure]["median"],
                 "low": result["intervals"][measure]["low"],
                 "high": result["intervals"][measure]["high"]}
           for key, measure in (("mdl", "normalized_mdl"), ("d90", "d90"), ("bytes", "compressed_bytes"))}}


def _legacy_exploratory_correlations(root: Path, mdl_paths, id_paths, compression_paths, llc_paths) -> dict | None:
    from scipy.stats import spearmanr
    cells: dict[str, dict[str, np.ndarray]] = {}
    for path in mdl_paths:
        key = _rel_cell(path, root)
        if "/cells/" not in f"/{key}/" or not key.endswith(".last") or path.stem != "mdl.mlp": continue
        frame = read_table(path); frame = frame[frame.control == "observed"]
        cells.setdefault(key, {})["mdl"] = (frame.total_bits / frame.marginal_bits).to_numpy()
    for path in id_paths:
        key = _rel_cell(path, root); frame = read_table(path)
        hits = frame[frame.is_d90.astype(str).str.lower().isin(["true", "1"])]
        if len(hits): cells.setdefault(key, {})["d90"] = hits.dimension.to_numpy(dtype=float)
    for path in compression_paths:
        key = _rel_cell(path, root); frame = read_table(path)
        ok = frame[frame.acceptable.astype(str).str.lower().isin(["true", "1"])]
        if len(ok): cells.setdefault(key, {})["bytes"] = np.array([ok.bytes.min()], dtype=float)
    for path in llc_paths:
        key = _rel_cell(path, root); frame = read_table(path)
        accepted = frame.accepted.astype(str).str.lower().isin(["true", "1"])
        central = frame[(frame.lr_multiplier == 1) & (frame.localization_multiplier == 1) & accepted]
        rejected = central.estimate_rejected.astype(str).str.lower().isin(["true", "1"])
        if len(central) and not rejected.any():
            cells.setdefault(key, {})["llc"] = central.llc.to_numpy(dtype=float)
    complete = {k: v for k, v in cells.items() if all(m in v for m in ("mdl", "d90", "bytes", "llc"))}
    if len(complete) < 5: return None
    keys = list(complete); rng = np.random.default_rng(2026); result = {"n_cells": len(keys)}
    for measure in ("mdl", "d90", "bytes"):
        central_x = [np.mean(complete[k]["llc"]) for k in keys]
        central_y = [np.mean(complete[k][measure]) for k in keys]
        estimate = float(spearmanr(central_x, central_y).statistic)
        boot = []
        for _ in range(2000):
            sampled = rng.integers(0, len(keys), len(keys))
            xs, ys = [], []
            for i in sampled:
                cell = complete[keys[i]]
                xs.append(rng.choice(cell["llc"])); ys.append(rng.choice(cell[measure]))
            rho = spearmanr(xs, ys).statistic
            if np.isfinite(rho): boot.append(rho)
        result[measure] = {"rho": estimate, "low": float(np.quantile(boot, .025)),
                           "high": float(np.quantile(boot, .975))}
    result["convergent"] = all(result[m]["low"] > 0 for m in ("mdl", "d90", "bytes"))
    return result


def build_report(cfg: dict, root: Path) -> str:
    lines = ["# Model-relative complexity of KNOWS", "",
        "This report concerns compact decoding of each model's own forced-choice decision rule. "
        "It does not estimate the absolute Kolmogorov complexity of the human concept of knowledge.", ""]
    scenarios_path = root / "scenarios.parquet"
    if scenarios_path.exists() or scenarios_path.with_suffix(".tsv").exists():
        scenarios = read_table(scenarios_path)
        counts = scenarios.groupby("split").scenario_id.nunique().to_dict()
        lines += ["## Protocol audit", "", f"Underlying scenarios: {scenarios.scenario_id.nunique():,}; "
                  f"surface realizations: {len(scenarios):,}; grouped split counts: `{counts}`.", ""]
    label_paths = _tables(root, "labels")
    sample_eligible = {}
    lines += ["## Label stability", ""]
    if not label_paths:
        lines += ["_Labels have not been run._", ""]
    else:
        lines += ["| Cell | Predicate | Stable | Total | Class 0 | Class 1 | Sample-count gate |",
                  "|:--|:--|--:|--:|--:|--:|:--|"]
        for path in label_paths:
            frame = read_table(path)
            for pred, group in frame.groupby("predicate"):
                stable = group[group.stable.astype(str).str.lower().isin(["true", "1"])]
                classes = stable.label.astype(str).str.lower().value_counts()
                c0 = int(classes.get("false", 0) + classes.get("0", 0) + classes.get("0.0", 0))
                c1 = int(classes.get("true", 0) + classes.get("1", 0) + classes.get("1.0", 0))
                eligible = min(c0, c1) >= int(cfg["thresholds"]["minimum_stable_per_class"])
                sample_eligible[(path.parent.name, pred)] = eligible
                lines.append(f"| {_rel_cell(path, root)} | {pred} | {len(stable)} | {len(group)} | {c0} | {c1} | {'yes' if eligible else 'no'} |")
        lines.append("")
        primary = next((p for p in label_paths if cfg["models"]["primary"].split("/")[-1].lower() in str(p).lower()), label_paths[0])
        labels = read_table(primary)
        unstable = labels[~labels.stable.astype(str).str.lower().isin(["true", "1"])].head(10)
        lines += ["### Representative ambiguity cases", ""]
        if len(unstable):
            scenarios = read_table(scenarios_path)[["example_id", "scenario_text"]]
            cases = unstable.merge(scenarios, on="example_id", how="left")
            for row in cases.itertuples():
                lines.append(f"- `{row.example_id}` / **{row.predicate}**, confidence {float(row.confidence):.3f}: {row.scenario_text}")
            lines.append("")
        else: lines += ["_No unstable cases._", ""]
    mdl_paths = _tables(root, "mdl.mlp") + _tables(root, "mdl.linear")
    lines += ["## Layerwise online MDL", ""]
    if not mdl_paths:
        lines += ["_MDL has not been run._", ""]
    else:
        lines += ["| Cell | Probe | Bits/label | Compression | Eval macro-F1 | Within-predicate quality gate |",
                  "|:--|:--|--:|--:|--:|:--|"]
        for path in mdl_paths:
            frame = read_table(path)
            observed = frame[frame.control == "observed"]
            f1 = observed.macro_f1.mean() if "macro_f1" in observed else np.nan
            parts = path.relative_to(root).parts
            eligible = (len(parts) >= 5 and parts[1] == "cells" and
                        sample_eligible.get((parts[0], parts[2]), False) and
                        f1 >= float(cfg["thresholds"]["minimum_macro_f1"]))
            probe = "mlp" if "mlp" in path.name else "linear"
            lines.append(f"| {_rel_cell(path, root)} | {probe} | {observed.bits_per_label.mean():.3f} | {observed.compression.mean():.3f} | {f1:.3f} | {'yes' if eligible else 'no'} |")
        lines.append("")
    id_paths = _tables(root, "intrinsic_dimension")
    lines += ["## Random-subspace dimension", ""]
    if id_paths:
        lines += ["| Cell | Median d90 | Projection seeds reaching target |", "|:--|--:|--:|"]
        for path in id_paths:
            frame = read_table(path); hits = frame[frame.is_d90.astype(str).str.lower().isin(["true", "1"])]
            lines.append(f"| {_rel_cell(path, root)} | {hits.dimension.median() if len(hits) else 'not reached'} | {hits.projection_seed.nunique()} |")
        lines.append("")
    else: lines += ["_Intrinsic dimension has not been run._", ""]
    compression_paths = _tables(root, "compression")
    lines += ["## Practical probe compression", ""]
    if compression_paths:
        lines += ["| Cell | Tuning-selected bytes (median across seeds) | Selected test macro-F1 |", "|:--|--:|--:|"]
        for path in compression_paths:
            frame = read_table(path)
            ok = frame[frame.selected.astype(str).str.lower().isin(["true", "1"])] if "selected" in frame else frame.iloc[:0]
            critical = int(ok.bytes.median()) if len(ok) else "not reached"
            f1 = f"{ok.test_macro_f1.mean():.3f}" if len(ok) and "test_macro_f1" in ok else "—"
            lines.append(f"| {_rel_cell(path, root)} | {critical} | {f1} |")
        lines.append("")
    else: lines += ["_Compression has not been run._", ""]
    llc_paths = _tables(root, "llc")
    lines += ["## LLC diagnostics and convergence", ""]
    if llc_paths:
        lines += ["| Cell | Central LLC | Rejected | Diagnostic |", "|:--|--:|:--|:--|"]
        for path in llc_paths:
            frame = read_table(path); central = frame[(frame.lr_multiplier == 1) & (frame.localization_multiplier == 1)]
            lines.append(f"| {_rel_cell(path, root)} | {central.llc.mean():.3f} | {central.estimate_rejected.iloc[0]} | {central.rejection_reason.iloc[0]} |")
        lines.append("")
        lines += _calibration_status(root, cfg)
    else: lines += ["_LLC is staged and has not been run._", ""]
    convergence = _convergence(root, mdl_paths, id_paths, compression_paths, llc_paths)
    lines += ["### Cross-measure directional tests", ""]
    if convergence:
        lines += ["| LLC versus | Median bootstrap Spearman rho | 95% bootstrap interval |", "|:--|--:|:--|"]
        for key, label in (("mdl", "normalized MDL"), ("d90", "d90"), ("bytes", "critical bytes")):
            value = convergence[key]
            lines.append(f"| {label} | {value['rho']:.3f} | [{value['low']:.3f}, {value['high']:.3f}] |")
        lines += ["", f"Convergent under the preregistered all-three criterion: **{'yes' if convergence['convergent'] else 'no'}** "
                  f"({convergence['n_cells']} complete cells).", ""]
    else:
        lines += ["_Confirmatory correlations are unavailable: no complete, accepted 15-cell paired-refit "
                  "bootstrap result exists. Seed/chain resampling alone does not meet the preregistered "
                  "scenario-uncertainty requirement._", ""]
    lines += ["## Interpretation guardrails", "",
        "Per-predicate tables condition on that predicate's stable subset. Passing their quality gate "
        "does not establish cross-predicate comparability; cross-predicate inference requires the common "
        "stable cohort used by the paired-refit analysis.", "",
        "Comparative claims are suppressed above when either stable class has fewer than the configured minimum "
        "or untouched-evaluation macro-F1 is below 0.75. MDL label codelength, constructive subspace dimension, "
        "serialized probe size, and LLC are reported as distinct measurements; agreement is an empirical result, not an assumption.", ""]
    text = "\n".join(lines)
    (root / "REPORT.md").write_text(text)
    return text
