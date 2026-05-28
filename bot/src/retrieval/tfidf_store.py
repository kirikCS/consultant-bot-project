"""TF-IDF индекс по символьным n-граммам (3–5), устойчивый к опечаткам и транслитерации."""
from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel


class TfidfStore:
    def __init__(self, vectorizer: TfidfVectorizer, doc_matrix: Any) -> None:
        self._vec = vectorizer
        self._mat = doc_matrix

    @classmethod
    def load(cls, indices_dir: str) -> "TfidfStore":
        path = Path(indices_dir) / "tfidf.pkl"
        with open(path, "rb") as f:
            blob = pickle.load(f)
        return cls(blob["vectorizer"], blob["matrix"])

    def search(self, query: str, k: int) -> list[tuple[int, float]]:
        if not query or not query.strip():
            return []
        q_vec = self._vec.transform([query])
        sims = linear_kernel(q_vec, self._mat).ravel()
        if k >= len(sims):
            idxs = np.argsort(-sims)
        else:
            top = np.argpartition(-sims, k)[:k]
            idxs = top[np.argsort(-sims[top])]
        return [(int(i), float(sims[i])) for i in idxs if sims[i] > 0.0]


def build_and_save(corpus: list[str], indices_dir: str) -> tuple[int, int]:
    vec = TfidfVectorizer(
        analyzer="char_wb",
        ngram_range=(3, 5),
        min_df=1,
        sublinear_tf=True,
        norm="l2",
    )
    mat = vec.fit_transform(corpus)

    out = Path(indices_dir) / "tfidf.pkl"
    with open(out, "wb") as f:
        pickle.dump(
            {"vectorizer": vec, "matrix": mat},
            f,
            protocol=pickle.HIGHEST_PROTOCOL,
        )
    return len(vec.vocabulary_), int(mat.nnz)
