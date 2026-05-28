"""7-вариантный сравнительный замер retriever-каналов: TF-IDF / BM25 / Dense + все парные и 3-way гибриды."""
from __future__ import annotations

import asyncio
import logging

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

from src.config import settings
from src.retrieval.bm25_store import BM25Store
from src.retrieval.catalog import Catalog
from src.retrieval.embedder import EmbedderClient
from src.retrieval.tokenize import tokenize
from src.retrieval.vector_store import VectorStore

QUERIES = [
    "ImmunoHealth",
    "ImmunoHealth 111",
    "ударно-волновая терапия",
    "болит живот",
    "постоянная усталость",
    "выпадают волосы",
    "болит сердце",
    "иммунохелс",
    "приём гастроэнтеролога",
    "невролог по головной боли",
    "анализы при болях в животе",
    "ферритин",
    "имунохэлс",
    "имуннохелс",
    "имуннохэлс",
    "иммонохелс",
]

TOP_N = 5
RRF_K = 60


def build_tfidf(corpus: list[str]):
    vec = TfidfVectorizer(
        analyzer="char_wb",
        ngram_range=(3, 5),
        min_df=1,
        sublinear_tf=True,
        norm="l2",
    )
    mat = vec.fit_transform(corpus)
    return vec, mat


def tfidf_search(vec, mat, query: str, k: int) -> list[tuple[int, float]]:
    q = vec.transform([query])
    sims = linear_kernel(q, mat).ravel()
    if k >= len(sims):
        idxs = np.argsort(-sims)
    else:
        top = np.argpartition(-sims, k)[:k]
        idxs = top[np.argsort(-sims[top])]
    return [(int(i), float(sims[i])) for i in idxs if sims[i] > 0.0]


def rrf(*ranked_lists: list[tuple[int, float]], top: int, k: int = RRF_K) -> list[int]:
    scores: dict[int, float] = {}
    for lst in ranked_lists:
        for rank, (row, _) in enumerate(lst):
            scores[row] = scores.get(row, 0.0) + 1.0 / (k + rank + 1)
    return [r for r, _ in sorted(scores.items(), key=lambda kv: -kv[1])[:top]]


def fmt(rows: list[dict]) -> str:
    if not rows:
        return "  (none)"
    return "\n".join(
        f"  {i+1}. [{p['category'][:14]:14}] {p['name'][:88]}"
        for i, p in enumerate(rows)
    )


async def main() -> None:
    logging.basicConfig(level=logging.WARNING)

    print("Loading FAISS + BM25 + meta…")
    vs = VectorStore.load(settings.indices_dir)
    bm25 = BM25Store.load(settings.indices_dir)
    catalog = Catalog.load(settings.indices_dir)
    embedder = EmbedderClient()

    print("Loading meta corpus for TF-IDF…")
    meta = pd.read_parquet(f"{settings.indices_dir}/meta.parquet")
    corpus = meta["text"].astype(str).tolist()
    print(f"  {len(corpus)} docs")

    print("Building TF-IDF char-3-5 gram index…")
    tfidf_vec, tfidf_mat = build_tfidf(corpus)
    print(f"  vocab={len(tfidf_vec.vocabulary_)}, nnz={tfidf_mat.nnz}")

    try:
        for q in QUERIES:
            print("=" * 110)
            print(f"QUERY: {q!r}")
            print("=" * 110)

            bm25_hits = bm25.search(tokenize(q), 30)
            tfidf_hits = tfidf_search(tfidf_vec, tfidf_mat, q, 30)
            vec = await embedder.embed(q)
            dense_hits = vs.search(vec, 30)

            modes = {
                "TF-IDF only": [r for r, _ in tfidf_hits[:TOP_N]],
                "BM25 only": [r for r, _ in bm25_hits[:TOP_N]],
                "Dense only": [r for r, _ in dense_hits[:TOP_N]],
                "Hybrid BM25+TF-IDF": rrf(bm25_hits, tfidf_hits, top=TOP_N),
                "Hybrid BM25+Dense": rrf(bm25_hits, dense_hits, top=TOP_N),
                "Hybrid TF-IDF+Dense": rrf(tfidf_hits, dense_hits, top=TOP_N),
                "Hybrid all-3 (RRF)": rrf(bm25_hits, tfidf_hits, dense_hits, top=TOP_N),
            }

            for label, rows in modes.items():
                if not rows:
                    print(f"\n[{label}]\n  (none)")
                else:
                    items = catalog.many(rows)
                    print(f"\n[{label}]")
                    print(fmt(items))
            print()
    finally:
        await embedder.aclose()


if __name__ == "__main__":
    asyncio.run(main())
