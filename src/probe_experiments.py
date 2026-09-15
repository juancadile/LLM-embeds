"""CLI for the staged model-relative epistemic probing study.

Usage:
  python -m src.probe_experiments --config configs/knows_mdl.yaml \
      --stage labels|extract|mdl|id|compress|llc|report
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path

import numpy as np

from .probe_data import (PREDICATES, assign_splits, generate_scenarios,
                         validate_scenarios, validate_grouped_splits)
from .probe_utils import (atomic_json, config_diff, load_config, output_dir, read_table,
                         software_manifest, write_table, scenario_digest, config_digest)


def slug(value: str) -> str:
    return re.sub(r"[^a-z0-9.-]+", "-", value.lower()).strip("-")


def models(cfg: dict, selected: list[str] | None = None) -> list[str]:
    values = [cfg["models"]["primary"], *cfg["models"].get("replications", [])]
    if selected: values = [m for m in values if m in selected or slug(m) in selected]
    return values


def validate_predicates(cfg: dict) -> None:
    """Reject YAML coercions and unknown predicates before loading a model."""
    predicates = cfg.get("predicates", list(PREDICATES))
    if (not isinstance(predicates, list) or
        not all(isinstance(predicate, str) for predicate in predicates)):
        raise ValueError("predicates must be a list of strings; quote YAML-sensitive names such as 'true'")
    unknown = sorted(set(predicates).difference(PREDICATES))
    if unknown:
        raise ValueError(f"unknown predicates: {unknown}")
    if len(predicates) != len(set(predicates)):
        raise ValueError("predicates must not contain duplicates")


def scenario_frame(cfg: dict):
    root = output_dir(cfg); path = root / "scenarios.parquet"
    if path.exists() or path.with_suffix(".tsv").exists():
        cached = read_table(path)
        validate_scenarios(cached)
        validate_grouped_splits(cached)
        expected = assign_splits(generate_scenarios(int(cfg["data"]["n_scenarios"]),
            int(cfg["data"]["paraphrases"]), int(cfg["seed"]),
            int(cfg["data"].get("development_scenarios", 0)),
            cfg["data"].get("canonical_support_fraction")),
            cfg["data"]["split"], int(cfg["seed"]))
        if not cached.equals(expected):
            raise ValueError("scenario cache differs from current configuration; use a new output directory")
        return cached
    frame = generate_scenarios(int(cfg["data"]["n_scenarios"]), int(cfg["data"]["paraphrases"]),
                              int(cfg["seed"]), int(cfg["data"].get("development_scenarios", 0)),
                              cfg["data"].get("canonical_support_fraction"))
    frame = assign_splits(frame, cfg["data"]["split"], int(cfg["seed"]))
    write_table(frame, path)
    atomic_json({**software_manifest(), "seed": cfg["seed"], "factors": [
        "truth", "belief", "evidence_source", "reliable", "defeater", "gettier", "guessing"],
        "split_unit": "scenario_id", "paraphrases_grouped": True,
        "canonical_support_fraction": cfg["data"].get("canonical_support_fraction")},
        root / "scenarios.sidecar.json")
    return frame


def labels_with_rows(root: Path, model_id: str, predicate: str):
    import json
    model_root = root / slug(model_id)
    frame = read_table(root / "scenarios.parquet")
    digest = scenario_digest(frame)
    label_meta = json.loads((model_root / "labels.sidecar.json").read_text())
    activation_meta = json.loads((model_root / "activations.sidecar.json").read_text())
    for meta in (label_meta, activation_meta):
        if meta.get("scenario_digest") != digest or meta.get("row_order") != frame.example_id.tolist():
            raise ValueError("artifact provenance or activation row order mismatch")
    if label_meta.get("requested_revision") != activation_meta.get("requested_revision"):
        raise ValueError("labels and representations used different model revisions")
    if label_meta.get("config_digest") != activation_meta.get("config_digest"):
        raise ValueError("labels and representations used different configurations")
    labels = read_table(root / slug(model_id) / "labels.parquet")
    labels = labels[labels.predicate == predicate].copy()
    scenarios = read_table(root / "scenarios.parquet").reset_index().rename(columns={"index": "activation_row"})
    return labels.drop(columns=["split"], errors="ignore").merge(
        scenarios[["example_id", "scenario_id", "split", "activation_row"] +
                  [c for c in ("minimal_pair_group", "split_group") if c in scenarios]],
        on=["example_id", "scenario_id"], validate="one_to_one")


def cell_dir(root: Path, model_id: str, predicate: str, layer: str, pool: str) -> Path:
    return root / slug(model_id) / "cells" / predicate / f"{layer}.{pool}"


def llc_gate(cell: Path, cfg: dict) -> tuple[bool, str]:
    required = [cell / "mdl.mlp.parquet", cell / "intrinsic_dimension.parquet",
                cell / "compression.parquet"]
    if any(not p.exists() and not p.with_suffix(".tsv").exists() for p in required):
        return False, "experiments_1_to_3_incomplete"
    mdl = read_table(required[0]); observed = mdl[mdl.control == "observed"]
    shuffled = mdl[mdl.control == "shuffled_labels"]
    if "macro_f1" not in observed or observed.macro_f1.mean() < float(cfg["thresholds"]["minimum_macro_f1"]):
        return False, "macro_f1_below_threshold"
    if observed.total_bits.mean() > (1 - float(cfg["thresholds"]["mdl_shuffled_improvement"])) * shuffled.total_bits.mean():
        return False, "mdl_control_not_passed"
    dimension = read_table(required[1])
    expected_projection_seeds = set(cfg["intrinsic_dimension"]["projection_seeds"])
    hits = dimension[dimension["is_d90"].astype(str).str.lower().isin(["true", "1"])]
    if (hits.empty or set(hits["projection_seed"].astype(int)) != expected_projection_seeds or
            hits.groupby("projection_seed").size().ne(1).any()):
        return False, "intrinsic_dimension_not_reached"
    compression = read_table(required[2])
    expected_probe_seeds = set(cfg["seeds"])
    selected = compression[compression["selected"].astype(str).str.lower().isin(["true", "1"])]
    if (selected.empty or set(selected["seed"].astype(int)) != expected_probe_seeds or
            selected.groupby("seed").size().ne(1).any() or
            not selected["acceptable"].astype(str).str.lower().isin(["true", "1"]).all()):
        return False, "compression_criterion_not_met"
    return True, "passed"


def accept_amendment(root: Path, recorded: dict, cfg: dict, digest: str) -> dict:
    """A mid-experiment configuration change must be declared before it is used.

    The declaration has to chain from the digest actually in force and enumerate
    the exact settings it changes, so no edit can pass unnamed.
    """
    import json
    declared = root / "config_amendments.json"
    if not declared.exists():
        raise ValueError("configuration changed within an experiment and no amendment is declared; "
                         "use a new output directory or declare the change in config_amendments.json")
    entries = json.loads(declared.read_text())
    entry = entries[-1] if entries else {}
    changed = config_diff(recorded.get("configuration", {}), cfg)
    if entry.get("previous_digest") != recorded.get("config_digest") or entry.get("new_digest") != digest:
        raise ValueError("declared amendment does not chain from the configuration in force")
    if sorted(entry.get("changed_settings", [])) != changed:
        raise ValueError(f"declared amendment names {sorted(entry.get('changed_settings', []))}, "
                         f"but the configuration changes {changed}")
    if not str(entry.get("reason", "")).strip():
        raise ValueError("a declared amendment must record why the configuration changed")
    return {**entry, "changed_settings": changed}


def run(args) -> None:
    cfg = load_config(args.config)
    validate_predicates(cfg)
    root = output_dir(cfg); root.mkdir(parents=True, exist_ok=True)
    import json
    manifest = root / "experiment.sidecar.json"
    digest = config_digest(cfg)
    if manifest.exists():
        recorded = json.loads(manifest.read_text())
        if recorded.get("config_digest") != digest:
            amendment = accept_amendment(root, recorded, cfg, digest)
            atomic_json({**recorded, "config_digest": digest, "configuration": cfg,
                         "amendments": [*recorded.get("amendments", []), amendment],
                         **software_manifest()}, manifest)
    else:
        atomic_json({"config_digest": digest, "configuration": cfg,
                     **software_manifest()}, manifest)
    frame = scenario_frame(cfg)
    chosen_models = models(cfg, args.model)
    if args.stage == "labels":
        from .probe_hf import label_scenarios
        for model_id in chosen_models: label_scenarios(frame, model_id, cfg, root / slug(model_id))
    elif args.stage == "extract":
        from .probe_hf import extract_scenarios
        for model_id in chosen_models: extract_scenarios(frame, model_id, cfg, root / slug(model_id))
    elif args.stage in ("mdl", "id"):
        predicates = args.predicate or cfg["predicates"]
        layers = args.layer or cfg["extract"]["layers"]
        pools = args.pool or cfg["extract"]["pooling"]
        for model_id in chosen_models:
            for predicate in predicates:
                labels = labels_with_rows(root, model_id, predicate)
                if args.stage == "mdl":
                    from .probe_controls import run_surface_controls
                    run_surface_controls(labels, frame, int(cfg["seed"]),
                                         root / slug(model_id) / "controls" / predicate)
                for layer in layers:
                    for pool in pools:
                        activation_path = root / slug(model_id) / f"activations.{layer}.{pool}.npy"
                        x = np.load(activation_path, mmap_mode="r")
                        out = cell_dir(root, model_id, predicate, layer, pool)
                        if args.stage == "mdl":
                            from .probe_mdl import run_mdl_cell
                            for kind in cfg["probe"]["kinds"]:
                                run_mdl_cell(x, labels, kind, cfg["probe"], cfg["seeds"], out,
                                    {"model_id": model_id, "predicate": predicate, "layer": layer, "pool": pool})
                                qpath = root / slug(model_id) / f"control.question_only.{predicate}.{layer}.{pool}.npy"
                                if qpath.exists():
                                    qout = root / slug(model_id) / "controls" / predicate / f"question_only.{layer}.{pool}"
                                    run_mdl_cell(np.load(qpath, mmap_mode="r"), labels, kind, cfg["probe"],
                                        cfg["seeds"], qout, {"model_id": model_id, "predicate": predicate,
                                        "layer": layer, "pool": pool, "control": "question_only"})
                        else:
                            from .probe_id import run_id_cell
                            run_id_cell(x, labels, cfg["probe"], cfg["intrinsic_dimension"], out)
        # Preregistered escalation: only Qwen3-14B KNOWS receives a complete block
        # sweep, and only after one of the five primary last-token depths clears both gates.
        if args.stage == "mdl" and cfg["models"]["primary"] in chosen_models and "knows" in predicates:
            primary = cfg["models"]["primary"]
            passed = False
            for layer in cfg["extract"]["layers"]:
                p = cell_dir(root, primary, "knows", layer, "last") / "mdl.mlp.parquet"
                if not p.exists() and not p.with_suffix(".tsv").exists(): continue
                table = read_table(p); observed = table[table.control == "observed"]
                shuffled = table[table.control == "shuffled_labels"]
                passed |= ("tune_macro_f1" in observed and observed.tune_macro_f1.mean() >= float(cfg["thresholds"]["minimum_macro_f1"]) and
                           observed.total_bits.mean() <= (1 - float(cfg["thresholds"]["mdl_shuffled_improvement"])) * shuffled.total_bits.mean())
            if passed:
                labels = labels_with_rows(root, primary, "knows")
                base = root / slug(primary)
                for activation_path in sorted(base.glob("activations.block*.last.npy")):
                    layer = activation_path.name.split(".")[1]
                    out = cell_dir(root, primary, "knows", layer, "last")
                    for kind in cfg["probe"]["kinds"]:
                        run_mdl_cell(np.load(activation_path, mmap_mode="r"), labels, kind,
                            cfg["probe"], cfg["seeds"], out,
                            {"model_id": primary, "predicate": "knows", "layer": layer,
                             "pool": "last", "preregistered_full_layer_sweep": True})
    elif args.stage == "compress":
        from .probe_compress import run_compression
        for model_id in chosen_models:
            for predicate in (args.predicate or cfg["predicates"]):
                labels = labels_with_rows(root, model_id, predicate)
                stable = labels[labels.stable.astype(str).str.lower().isin(["true", "1"])]
                test = stable.split == "evaluation"
                tune = stable.split == "tune"
                for layer in (args.layer or cfg["extract"]["layers"]):
                    for pool in (args.pool or cfg["extract"]["pooling"]):
                        out = cell_dir(root, model_id, predicate, layer, pool)
                        x = np.load(root / slug(model_id) / f"activations.{layer}.{pool}.npy", mmap_mode="r")
                        results = []
                        for seed in cfg["seeds"]:
                            probe = out / f"probe.mlp.seed{seed}.full.pt"
                            result = run_compression(probe, x[stable.loc[tune, "activation_row"].astype(int)],
                                stable.loc[tune, "label"].astype(int).to_numpy(), cfg["compression"],
                                out / "compression_seeds" / str(seed),
                                x_test=x[stable.loc[test, "activation_row"].astype(int)],
                                y_test=stable.loc[test, "label"].astype(int).to_numpy())
                            result["seed"] = seed
                            results.append(result)
                        import pandas as pd
                        write_table(pd.concat(results, ignore_index=True), out / "compression.parquet")
    elif args.stage == "llc":
        from .probe_llc import run_llc, validate_regular_logistic
        validation_path = root / "llc_logistic_validation.json"
        # Revalidate the executing sampler; a stale JSON pass is not evidence.
        validation = validate_regular_logistic(int(cfg["seed"]))
        atomic_json(validation, validation_path)
        if not validation["passed"]:
            raise RuntimeError("LLC sampler failed the regular logistic d/2 validation; refusing MLP LLC runs")
        primary = cfg["models"]["primary"]
        if primary not in chosen_models: return
        from .probe_calibration import calibrate
        candidates = []
        cells = root / slug(primary) / "cells" / "knows"
        for folder in cells.glob("*.last"):
            p = folder / "mdl.mlp.parquet"
            if p.exists() or p.with_suffix(".tsv").exists():
                m = read_table(p)
                candidates.append((m[m.control == "observed"].bits_per_label.mean(), folder.name[:-5]))
        if not candidates:
            raise ValueError("KNOWS MDL must run before LLC calibration")
        best = min(candidates)[1]
        calibration_cell = cell_dir(root, primary, "knows", best, "last")
        passed, reason = llc_gate(calibration_cell, cfg)
        if not passed:
            raise ValueError(f"KNOWS calibration cell failed prerequisite gate: {reason}")
        calibration_labels = labels_with_rows(root, primary, "knows")
        selected = calibration_labels[calibration_labels.stable & (calibration_labels.split == "coding")]
        calibration_x = np.load(root / slug(primary) / f"activations.{best}.last.npy", mmap_mode="r")
        frozen = calibrate(calibration_cell / f"probe.mlp.seed{cfg['seeds'][0]}.full.pt",
            calibration_x[selected.activation_row.astype(int)], selected.label.astype(int).to_numpy(),
            cfg["llc"], cfg["seeds"], root / slug(primary) / "llc_calibration")
        # Freeze one KNOWS-selected layer and one calibration for the comparison grid.
        for predicate in (args.predicate or cfg["predicates"]):
            labels = labels_with_rows(root, primary, predicate)
            stable = labels[labels.stable.astype(str).str.lower().isin(["true", "1"])]
            train = stable.split == "coding"
            for layer in dict.fromkeys(["emb", best, "final"]):
                out = cell_dir(root, primary, predicate, layer, "last")
                passed, reason = llc_gate(out, cfg)
                if not passed:
                    atomic_json({"stage": "llc", "skipped": True, "reason": reason}, out / "llc.sidecar.json")
                    print(f"[llc] skip {predicate}/{layer}: {reason}", flush=True)
                    continue
                x = np.load(root / slug(primary) / f"activations.{layer}.last.npy", mmap_mode="r")
                idx = stable.loc[train, "activation_row"].astype(int).to_numpy()
                run_llc(out / f"probe.mlp.seed{cfg['seeds'][0]}.full.pt", x[idx],
                        stable.loc[train, "label"].astype(int).to_numpy(), frozen, cfg["seeds"], out)
    elif args.stage == "report":
        if getattr(args, "bootstrap_draws", 0):
            from .probe_joint_refits import run_joint_refits
            run_joint_refits(cfg, root, args.bootstrap_draws)
        from .probe_report import build_report
        print(build_report(cfg, root))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--stage", required=True, choices=["labels", "extract", "mdl", "id", "compress", "llc", "report"])
    parser.add_argument("--model", action="append", help="model id or slug; repeat to select")
    parser.add_argument("--predicate", action="append")
    parser.add_argument("--layer", action="append")
    parser.add_argument("--pool", action="append")
    parser.add_argument("--bootstrap-draws", type=int, default=0,
                        help="report stage: expensive paired refits of all four estimators; >=1000 for inference")
    run(parser.parse_args())


if __name__ == "__main__": main()
