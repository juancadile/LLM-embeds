"""Vectorised cosine-similarity tools: most_similar, define(a+b), define2.

``define2`` is exact over all C(n,2) word pairs: with s = V·t, Gram G = V·Vᵀ and row
norms n,  cos(v_i + v_j, t) = (s_i + s_j) / (|t| · sqrt(n_i² + n_j² + 2 G_ij)).
For 5,124 words that is a 26M-entry matrix, computed in well under a second.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

MODES = ("raw", "center", "abtt")


def center(V: np.ndarray) -> np.ndarray:
    return V - V.mean(axis=0, keepdims=True)


def all_but_the_top(V: np.ndarray, D: int | None = None) -> np.ndarray:
    """Mu & Viswanath (2018): mean-centre, then remove the top-D principal directions.
    D defaults to dim // 100 as in the paper."""
    if D is None:
        D = max(1, V.shape[1] // 100)
    X = center(V).astype(np.float64)
    # top-D right singular vectors of the centred matrix
    _, _, Vt = np.linalg.svd(X, full_matrices=False)
    U = Vt[:D]                      # (D, dim)
    X = X - (X @ U.T) @ U
    return X.astype(np.float32)


def transform(V: np.ndarray, mode: str) -> np.ndarray:
    if mode == "raw":
        return V
    if mode == "center":
        return center(V)
    if mode == "abtt":
        return all_but_the_top(V)
    raise ValueError(mode)


def unit(X: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(X, axis=-1, keepdims=True)
    return X / np.maximum(n, 1e-12)


@dataclass
class Space:
    """An embedding matrix plus cached quantities for fast queries."""
    vocab: list[str]
    V: np.ndarray                                   # (n, d) float32, possibly transformed
    name: str = ""
    mode: str = "raw"
    idx: dict[str, int] = field(init=False)
    norms: np.ndarray = field(init=False)
    Vn: np.ndarray = field(init=False)
    _G: np.ndarray | None = field(default=None, init=False, repr=False)

    def __post_init__(self):
        self.idx = {w: i for i, w in enumerate(self.vocab)}
        self.norms = np.linalg.norm(self.V, axis=1).astype(np.float32)
        self.Vn = unit(self.V)

    @classmethod
    def from_raw(cls, vocab, V, name="", mode="raw"):
        return cls(vocab, transform(V, mode), name=name, mode=mode)

    def __contains__(self, w):
        return w in self.idx

    def vec(self, w: str) -> np.ndarray:
        return self.V[self.idx[w]]

    @property
    def G(self) -> np.ndarray:
        if self._G is None:
            self._G = (self.V @ self.V.T).astype(np.float32)
        return self._G

    # ---- single-vector queries ------------------------------------------------
    def cosines(self, q: np.ndarray) -> np.ndarray:
        qn = q / max(np.linalg.norm(q), 1e-12)
        return self.Vn @ qn.astype(np.float32)

    def most_similar(self, q: np.ndarray, k: int = 10, exclude=()) -> list[tuple[str, float]]:
        s = self.cosines(q).copy()
        for w in exclude:
            if w in self.idx:
                s[self.idx[w]] = -np.inf
        top = np.argpartition(-s, min(k, len(s) - 1))[:k]
        top = top[np.argsort(-s[top])]
        return [(self.vocab[i], float(s[i])) for i in top]

    def rank_of(self, q: np.ndarray, target: str, exclude=()) -> tuple[int, float]:
        """1-based rank of *target* among vocab words (excluding *exclude*) by cosine to q."""
        s = self.cosines(q).copy()
        t = self.idx[target]
        for w in exclude:
            if w in self.idx and w != target:
                s[self.idx[w]] = -np.inf
        rank = int((s > s[t]).sum()) + 1
        return rank, float(s[t])

    def define_ab(self, a: str, b: str, k: int = 10, exclude_ab: bool = False):
        q = self.vec(a) + self.vec(b)
        return self.most_similar(q, k, exclude=(a, b) if exclude_ab else ())

    # ---- all-pairs search ----------------------------------------------------
    def define2(self, t: np.ndarray, k: int = 10, exclude_idx=()) -> list[tuple[str, str, float]]:
        """Top-k pairs (i<j) maximising cos(v_i + v_j, t), skipping excluded rows."""
        n = len(self.vocab)
        s = (self.V @ t.astype(np.float32)).astype(np.float32)
        tn = float(np.linalg.norm(t))
        n2 = self.norms ** 2
        num = s[:, None] + s[None, :]
        den = np.sqrt(np.maximum(n2[:, None] + n2[None, :] + 2.0 * self.G, 1e-12)) * tn
        C = num / den
        C[np.tril_indices(n)] = -np.inf          # keep i<j only
        ex = np.fromiter(exclude_idx, dtype=np.int64)
        if ex.size:
            C[ex, :] = -np.inf
            C[:, ex] = -np.inf
        flat = C.ravel()
        top = np.argpartition(-flat, k)[:k]
        top = top[np.argsort(-flat[top])]
        out = []
        for f in top:
            i, j = divmod(int(f), n)
            out.append((self.vocab[i], self.vocab[j], float(flat[f])))
        return out

    def define2_word(self, target: str, k: int = 10, exclude_idx=()):
        ex = set(exclude_idx) | {self.idx[target]}
        return self.define2(self.vec(target), k, ex)
