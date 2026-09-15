"""Deterministic sample-count gate for completed model-judgment artifacts."""
from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

from .probe_utils import atomic_json, read_table, software_manifest


def _as_true(values):
    return values.astype(str).str.strip().str.lower().isin({"true", "1", "1.0"})


def evaluate_label_gate(labels, minimum_stable_per_class: int) -> dict:
    """Return per-predicate stable class counts without selecting on a split."""
    if minimum_stable_per_class < 1:
        raise ValueError("minimum_stable_per_class must be positive")
    required = {"predicate", "label", "stable", "split"}
    missing = sorted(required.difference(labels.columns))
    if missing:
        raise ValueError(f"label artifact is missing columns: {missing}")

    stable = labels[_as_true(labels["stable"])].copy()
    stable["positive"] = _as_true(stable["label"])
    predicates = {}
    for predicate, all_rows in labels.groupby("predicate", sort=True):
        kept = stable[stable.predicate == predicate]
        positive = int(kept.positive.sum())
        negative = int(len(kept) - positive)
        by_split = {}
        for split, group in kept.groupby("split", sort=True):
            split_positive = int(group.positive.sum())
            by_split[str(split)] = {
                "stable": int(len(group)),
                "negative": int(len(group) - split_positive),
                "positive": split_positive,
            }
        predicates[str(predicate)] = {
            "total": int(len(all_rows)),
            "stable": int(len(kept)),
            "unstable": int(len(all_rows) - len(kept)),
            "negative": negative,
            "positive": positive,
            "minimum_stable_per_class": int(minimum_stable_per_class),
            "eligible": min(negative, positive) >= minimum_stable_per_class,
            "by_split": by_split,
        }
    return {"predicates": predicates}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--labels", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--minimum", required=True, type=int)
    parser.add_argument("--require", action="append", default=[])
    args = parser.parse_args()

    labels_path = Path(args.labels)
    source = labels_path if labels_path.exists() else labels_path.with_suffix(".tsv")
    labels = read_table(labels_path)
    result = evaluate_label_gate(labels, args.minimum)
    missing = sorted(set(args.require).difference(result["predicates"]))
    failed = sorted(p for p in args.require
                    if p in result["predicates"] and not result["predicates"][p]["eligible"])
    result.update({
        "required_predicates": sorted(args.require),
        "missing_required_predicates": missing,
        "failed_required_predicates": failed,
        "passed": not missing and not failed,
        "labels_path": str(source.resolve()),
        "labels_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "gate_source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        **software_manifest(),
    })
    atomic_json(result, args.output)
    for predicate, counts in result["predicates"].items():
        print(f"{predicate}: stable={counts['stable']} negative={counts['negative']} "
              f"positive={counts['positive']} eligible={counts['eligible']}", flush=True)
    if not result["passed"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
