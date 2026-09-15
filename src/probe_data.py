"""Factorial epistemic scenarios, leakage-safe prompts, labels, and grouped splits."""
from __future__ import annotations

import itertools
import re
from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd

from .probe_utils import stable_int

PREDICATES = ("knows", "believes", "true", "justified", "lucky_guessed")
SCENARIO_VERSION = "epistemic-v3"
FORBIDDEN_REPRESENTATION_TERMS = re.compile(
    r"\b(knows?|believes?|true|truth|justified|lucky|guessed?|answer|choice|yes|no)\b",
    re.IGNORECASE,
)

NAMES = ("Ari", "Blair", "Casey", "Devon", "Emery", "Frankie", "Gray", "Harper")
CLAIMS = (
    ("the museum closes at six", "the posted closing time"),
    ("the train leaves from platform four", "the departure platform"),
    ("the red key opens the cabinet", "which key opens the cabinet"),
    ("the meeting is in room twelve", "the meeting room"),
    ("the orchard gate is unlocked", "the state of the orchard gate"),
    ("the package is in the mailroom", "the package location"),
    ("the lake path is open", "the state of the lake path"),
    ("the blue jar contains sugar", "the contents of the blue jar"),
)
SOURCES = {
    "perception": ("made a direct observation concerning {object}", "observations of this kind were usually accurate"),
    "testimony": ("was told by a staff member about {object}", "the staff member is usually reliable"),
    "instrument": ("checked a calibrated display for {object}", "the display usually works correctly"),
    "memory": ("recalled an earlier check of {object}", "the memory was normally dependable"),
}
PARAPHRASE_TEMPLATES = (
    "{name} considered whether {claim}. {source}. {reliability}. {belief}. {truth}. {defeater}. {luck}.",
    "For {name}, the proposition under consideration was that {claim}. {truth}. {source}; {reliability}. {belief}. {luck}. {defeater}.",
    "The issue was whether {claim}. {source_cap}, and {reliability}. {name}'s resulting position was described as follows: {belief}. {truth}. {defeater}. {luck}.",
)


def _realize(row: dict, paraphrase: int) -> str:
    source, reliability = SOURCES[row["evidence_source"]]
    source = row["name"] + " " + source.format(object=row["object"]) + ", which appeared to support the proposition"
    reliability = reliability if row["reliable"] else "the source had a strong record of error"
    belief = f"{row['name']} accepted the claim" if row["belief"] else f"{row['name']} rejected the claim"
    truth = "Events matched the claim" if row["truth"] else "Events did not match the claim"
    defeater = "Contrary information was absent" if not row["defeater"] else "Credible contrary information was available but left unresolved"
    if row["guessing"]:
        luck = "The position was selected arbitrarily without using the source"
    elif row["gettier"]:
        luck = "The apparent support failed independently, while a separate coincidence made events match"
    else:
        luck = "Separate coincidence was absent from the outcome"
    fields = {**row, "belief": belief, "truth": truth, "defeater": defeater}
    return PARAPHRASE_TEMPLATES[paraphrase].format(
        **fields, source=source, source_cap=source[:1].upper() + source[1:],
        reliability=reliability, luck=luck,
    )


def generate_scenarios(n_scenarios: int, paraphrases: int, seed: int,
                       development_scenarios: int = 0,
                       canonical_support_fraction: float | None = None) -> pd.DataFrame:
    """Sample constrained factorial cells, excluding all development families."""
    if paraphrases < 1 or paraphrases > len(PARAPHRASE_TEMPLATES):
        raise ValueError(f"paraphrases must be 1..{len(PARAPHRASE_TEMPLATES)}")
    factors = list(itertools.product(
        (False, True), (False, True), tuple(SOURCES), (False, True),
        (False, True), (False, True), (False, True),
    ))
    # A Gettier success presupposes an accepted, correct proposition supported
    # by evidence, rather than an arbitrary guess. These are structural
    # constraints, not independent binary factors.
    factors = [f for f in factors if not f[5] or (f[0] and f[1] and not f[6])]
    rng = np.random.default_rng(seed)
    candidates = list(itertools.product(factors, NAMES, CLAIMS))
    if not 1 <= n_scenarios <= len(candidates):
        raise ValueError(f"n_scenarios must be 1..{len(candidates)}")
    def pair_key(candidate):
        factor, name, (claim, _) = candidate
        _, belief, source, reliable, _defeater, gettier, guessing = factor
        return repr((name, claim, factor[0], belief, source, reliable, gettier, guessing))

    # Sample minimal-pair families as units.  Each family differs only in the
    # defeater factor, so both variants are available to the grouped splitter.
    families: dict[str, list[int]] = {}
    for i, candidate in enumerate(candidates):
        families.setdefault(pair_key(candidate), []).append(i)
    family_keys = list(families)
    if not 0 <= development_scenarios < len(family_keys):
        raise ValueError("invalid development exclusion size")
    family_order = rng.permutation(len(family_keys))
    excluded = {family_keys[i] for i in family_order[:development_scenarios]}
    eligible_families = [family_keys[i] for i in family_order
                         if family_keys[i] not in excluded]
    if canonical_support_fraction is not None:
        fraction = float(canonical_support_fraction)
        if not 0 <= fraction <= 1 or n_scenarios % 2:
            raise ValueError("canonical-support sampling requires an even scenario count and fraction in [0, 1]")
        target_scenarios = 2 * int(round(n_scenarios * fraction / 2))

        def is_canonical_support(key: str) -> bool:
            factor, _, _ = candidates[families[key][0]]
            truth, belief, _source, reliable, _defeater, gettier, guessing = factor
            return truth and belief and reliable and not gettier and not guessing

        support = [key for key in eligible_families if is_canonical_support(key)]
        controls = [key for key in eligible_families if not is_canonical_support(key)]
        support_count = target_scenarios // 2
        control_count = (n_scenarios - target_scenarios) // 2
        if len(support) < support_count or len(controls) < control_count:
            raise ValueError("insufficient families for requested canonical-support allocation")
        chosen = set(support[:support_count] + controls[:control_count])
        eligible_families = [key for key in eligible_families if key in chosen]
    eligible: list[int] = []
    for key in eligible_families:
        members = families[key]
        # Keep the two defeater variants adjacent and deterministic within a
        # family; shuffle their order only through the family ordering above.
        eligible.extend(sorted(members, key=lambda i: bool(candidates[i][0][4])))
        if len(eligible) >= n_scenarios:
            break
    if len(eligible) < n_scenarios:
        raise ValueError("insufficient scenarios after excluding development families")
    order = eligible[:n_scenarios]
    rows = []
    for sid, candidate in enumerate(order):
        factor, name, (claim, obj) = candidates[candidate]
        truth, belief, source, reliable, defeater, gettier, guessing = factor
        family = pair_key(candidates[candidate])
        base = dict(
            scenario_id=f"s{sid:05d}", name=name,
            minimal_pair_group=f"mp{stable_int(family):016x}",
            scenario_version=SCENARIO_VERSION,
            claim=claim, object=obj, truth=truth, belief=belief,
            evidence_source=source, reliable=reliable, defeater=defeater,
            gettier=gettier, guessing=guessing,
        )
        for p in range(paraphrases):
            rows.append({**base, "paraphrase_id": p, "example_id": f"s{sid:05d}-p{p}",
                         "scenario_text": _realize(base, p)})
    frame = pd.DataFrame(rows)
    validate_scenarios(frame)
    return frame


def validate_scenarios(frame: pd.DataFrame) -> None:
    if "scenario_version" not in frame or not frame.scenario_version.eq(SCENARIO_VERSION).all():
        raise ValueError("obsolete scenario artifact; use a new output directory")
    assert_no_representation_leakage(frame.scenario_text)
    if not all(r.claim in r.scenario_text for r in frame.itertuples()):
        raise ValueError("every paraphrase must contain its proposition")
    if (frame.gettier & (~frame.truth | ~frame.belief | frame.guessing)).any():
        raise ValueError("inconsistent Gettier factor combination")
    if frame.example_id.duplicated().any():
        raise ValueError("duplicate example ids")
    if frame.scenario_text.duplicated().any():
        raise ValueError("duplicate surface scenarios")


def assert_no_representation_leakage(texts: Iterable[str]) -> None:
    for i, text in enumerate(texts):
        match = FORBIDDEN_REPRESENTATION_TERMS.search(str(text))
        if match:
            raise ValueError(f"representation text {i} contains forbidden term {match.group()!r}")
        if "?" in str(text):
            raise ValueError(f"representation text {i} contains a question")


def assign_splits(frame: pd.DataFrame, split: dict[str, float], seed: int) -> pd.DataFrame:
    if not np.isclose(sum(split.values()), 1.0):
        raise ValueError("split fractions must sum to one")
    names = list(split)
    bounds = np.cumsum([split[k] for k in names])
    out = frame.copy()
    group_key = independent_groups(out)
    out["split_group"] = group_key
    mapping = {}
    for key in group_key.unique():
        u = stable_int(key, seed) / 2**64
        mapping[key] = names[min(np.searchsorted(bounds, u, side="right"), len(names) - 1)]
    out["split"] = group_key.map(mapping)
    validate_grouped_splits(out)
    return out


def independent_groups(frame: pd.DataFrame) -> pd.Series:
    """Connected components of shared scenario/minimal-pair identifiers.

    Joining identifiers as a composite key is insufficient: two different
    scenarios in the same minimal-pair family must belong to one component.
    Missing family identifiers do not connect otherwise unrelated scenarios.
    """
    parent = {}

    def find(key):
        parent.setdefault(key, key)
        if parent[key] != key:
            parent[key] = find(parent[key])
        return parent[key]

    for row in frame.itertuples(index=False):
        scenario = ("scenario", str(row.scenario_id))
        find(scenario)
        family = getattr(row, "minimal_pair_group", None)
        if pd.notna(family):
            a, b = find(scenario), find(("family", str(family)))
            parent[max(a, b)] = min(a, b)
    return pd.Series(["|".join(find(("scenario", str(s))))
                      for s in frame.scenario_id], index=frame.index)


def validate_grouped_splits(frame: pd.DataFrame) -> None:
    if (frame.groupby("scenario_id")["split"].nunique() > 1).any():
        raise AssertionError("paraphrases crossed splits")
    if "minimal_pair_group" in frame and (frame.groupby("minimal_pair_group")["split"].nunique() > 1).any():
        raise AssertionError("minimal-pair variants crossed splits")


QUESTIONS = {
    "knows": "Does {name} know that {claim}?",
    "believes": "Does {name} believe that {claim}?",
    "true": "Is it true that {claim}?",
    "justified": "Is {name} justified in accepting that {claim}?",
    "lucky_guessed": "Was {name}'s position a lucky guess?",
}


def forced_choice_prompt(row, predicate: str, reversed_order: bool) -> str:
    if predicate not in QUESTIONS:
        raise KeyError(predicate)
    def get(key):
        if hasattr(row, "_asdict"): return row._asdict()[key]
        if isinstance(row, dict): return row[key]
        return row[key]
    choices = "A. No\nB. Yes" if reversed_order else "A. Yes\nB. No"
    return ("Read the scenario and select exactly A or B.\n\nScenario:\n"
            f"{get('scenario_text')}\n\n{QUESTIONS[predicate].format(name=get('name'), claim=get('claim'))}\n"
            f"{choices}\nAnswer:")


def semantic_probability(logits: np.ndarray, reversed_order: bool) -> tuple[bool, float]:
    logits = np.asarray(logits, dtype=float)
    if logits.shape != (2,) or not np.isfinite(logits).all():
        raise ValueError("forced-choice scores must be two finite log probabilities")
    probs = np.exp(logits - logits.max()); probs /= probs.sum()
    yes_index = 1 if reversed_order else 0
    p_yes = float(probs[yes_index])
    return p_yes >= 0.5, max(p_yes, 1 - p_yes)


@dataclass
class LabelDecision:
    label: bool | None
    stable: bool
    confidence: float
    forward_label: bool
    reversed_label: bool


def combine_counterbalanced(forward_logits, reversed_logits, threshold: float) -> LabelDecision:
    a, pa = semantic_probability(forward_logits, False)
    b, pb = semantic_probability(reversed_logits, True)
    stable = a == b and min(pa, pb) >= threshold
    return LabelDecision(a if stable else None, stable, min(pa, pb), a, b)
