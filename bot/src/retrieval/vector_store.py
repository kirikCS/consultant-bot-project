"""Обёртка над сохранённым FAISS IndexFlatIP (плотные эмбеддинги услуг)."""
from __future__ import annotations

from pathlib import Path

import faiss
import numpy as np


class VectorStore:
    def __init__(self, index: faiss.Index) -> None:
        self._index = index

    @classmethod
    def load(cls, indices_dir: str) -> "VectorStore":
        path = Path(indices_dir) / "faiss.index"
        return cls(faiss.read_index(str(path)))

    def search(self, query_vec: np.ndarray, k: int) -> list[tuple[int, float]]:
        if query_vec.ndim == 1:
            query_vec = query_vec[None, :]
        q = query_vec.astype(np.float32)
        scores, idxs = self._index.search(q, k)
        return [(int(i), float(s)) for i, s in zip(idxs[0], scores[0]) if i >= 0]
