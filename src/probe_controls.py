"""Surface-only controls that never consume hidden representations."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression

from .probe_models import binary_metrics
from .probe_utils import write_table


def _style_features(texts) -> np.ndarray:
    rows = []
    for text in map(str, texts):
        tokens = text.split()
        rows.append([len(tokens), len(text), np.mean([len(t) for t in tokens]),
                     text.count("."), text.count(","), text.count(";")])
    return np.asarray(rows, dtype=np.float32)


def run_surface_controls(labels: pd.DataFrame, scenarios: pd.DataFrame, seed: int, out: Path) -> pd.DataFrame:
    data = labels[labels.stable.astype(str).str.lower().isin(["true", "1"])].merge(
        scenarios[["example_id", "scenario_text"]], on="example_id", validate="one_to_one")
    tr, te = data.split == "coding", data.split == "evaluation"
    y_train = data.loc[tr, "label"].astype(int); y_test = data.loc[te, "label"].astype(int)
    rows = []
    style_train, style_test = _style_features(data.loc[tr, "scenario_text"]), _style_features(data.loc[te, "scenario_text"])
    mean, sd = style_train.mean(0), style_train.std(0); sd[sd < 1e-8] = 1
    style = LogisticRegression(max_iter=1000, random_state=seed).fit((style_train - mean) / sd, y_train)
    rows.append({"control": "token_count_style", **binary_metrics(y_test, style.decision_function((style_test - mean) / sd))})
    vectorizer = CountVectorizer(binary=True, min_df=2, ngram_range=(1, 1))
    bag_train = vectorizer.fit_transform(data.loc[tr, "scenario_text"])
    bag = LogisticRegression(max_iter=1000, random_state=seed).fit(bag_train, y_train)
    rows.append({"control": "bag_of_tokens", **binary_metrics(y_test, bag.decision_function(vectorizer.transform(data.loc[te, "scenario_text"])))})
    result = pd.DataFrame(rows); out.mkdir(parents=True, exist_ok=True)
    write_table(result, out / "surface_controls.parquet")
    return result
