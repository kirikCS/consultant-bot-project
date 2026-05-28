"""Обёртка над сохранённым rank_bm25.BM25Okapi индексом каталога."""
from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np
from rank_bm25 import BM25Okapi


class BM25Store:
    def __init__(self, bm25: BM25Okapi) -> None:
        self._bm25 = bm25

    @classmethod
    def load(cls, indices_dir: str) -> "BM25Store":
        path = Path(indices_dir) / "bm25.pkl"
        with open(path, "rb") as f:
            blob = pickle.load(f)
        return cls(blob["bm25"])

    def search(self, query_tokens: list[str], k: int) -> list[tuple[int, float]]:
        if not query_tokens:
            return []
        scores = self._bm25.get_scores(query_tokens)
        if k >= len(scores):
            idxs = np.argsort(-scores)
        else:
            top = np.argpartition(-scores, k)[:k]
            idxs = top[np.argsort(-scores[top])]
        return [(int(i), float(scores[i])) for i in idxs if scores[i] > 0.0]
