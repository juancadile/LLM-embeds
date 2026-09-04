"""Load the cached embedding files.

Cache format: whitespace-separated floats, one row per line, no header and no word
column.  Row *i* is the vector of line *i* of the vocab file.  ``expanded_vocab.txt``
has 5,124 words; ``gpt_babbage.txt`` was produced against the older
``valid_vocab.txt`` (3,471 words) and must not be mixed with the others.
``gpt/gpt_ada-v2.txt`` is a git-LFS pointer whose object was never pushed; ignored.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .common import CACHE, ROOT, ensure_dirs

EXPANDED = "vocab/expanded_vocab.txt"
VALID = "vocab/valid_vocab.txt"

# name -> (embedding file, vocab file, expected dim, kind)
MODELS: dict[str, tuple[str, str, int, str]] = {
    "ada": ("gpt/gpt_ada.txt", EXPANDED, 1024, "openai-similarity-endpoint"),
    "babbage": ("gpt/gpt_babbage.txt", VALID, 2048, "openai-similarity-endpoint"),
    "curie": ("gpt/gpt_curie.txt", EXPANDED, 4096, "openai-similarity-endpoint"),
    "davinci": ("gpt/gpt_davinci.txt", EXPANDED, 12288, "openai-similarity-endpoint"),
    "opt-1.3b": ("opt/1_3B.txt", EXPANDED, 2048, "final-layer mean-pool, word alone"),
    "opt-13b": ("opt/13B.txt", EXPANDED, 5120, "final-layer mean-pool, word alone"),
    "t5-large": ("t5/t5large.txt", EXPANDED, 1024, "encoder final-layer mean-pool incl. EOS"),
    "t5-3b": ("t5/t53b.txt", EXPANDED, 1024, "encoder final-layer mean-pool incl. EOS"),
    "flan-t5-xxl": ("t5/flan_t5_11b.txt", EXPANDED, 4096, "encoder final-layer mean-pool incl. EOS"),
}

# Models used for the definition experiments (paper 3.2, second half) and the magnitude study.
DEFINE_MODELS = ["curie", "davinci", "opt-13b", "t5-3b", "flan-t5-xxl"]
MAGNITUDE_MODELS = ["opt-1.3b", "opt-13b", "t5-large", "t5-3b", "flan-t5-xxl"]


def load_vocab(path: str = EXPANDED) -> list[str]:
    words = [w.strip() for w in open(ROOT / path, encoding="utf-8")]
    words = [w for w in words if w]
    assert len(set(words)) == len(words), "duplicate words in vocab"
    return words


def load(name: str) -> tuple[list[str], np.ndarray]:
    """Return (vocab, float32 matrix) for a cached model, caching a .npy copy."""
    ensure_dirs()
    emb, vocab_path, dim, _ = MODELS[name]
    vocab = load_vocab(vocab_path)
    npy = CACHE / f"{name}.npy"
    if npy.exists():
        V = np.load(npy)
    else:
        df = pd.read_csv(ROOT / emb, sep=" ", header=None, dtype=np.float32).dropna(axis=1)
        V = np.ascontiguousarray(df.to_numpy(dtype=np.float32))
        np.save(npy, V)
    assert V.shape == (len(vocab), dim), f"{name}: got {V.shape}, expected {(len(vocab), dim)}"
    assert np.isfinite(V).all(), f"{name}: non-finite values"
    return vocab, V


def load_npy(path: Path, vocab_path: str = EXPANDED) -> tuple[list[str], np.ndarray]:
    """Load a matrix produced by src.extract (Phase 3)."""
    vocab = load_vocab(vocab_path)
    V = np.load(path).astype(np.float32)
    assert V.shape[0] == len(vocab), f"{path}: {V.shape[0]} rows vs {len(vocab)} words"
    return vocab, V
