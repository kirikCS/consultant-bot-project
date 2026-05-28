"""Сравнительный замер retriever-каналов: BM25-only / Dense-only / Hybrid (RRF)."""
from __future__ import annotations

import asyncio
import logging

from src.config import settings
from src.retrieval.bm25_store import BM25Store
from src.retrieval.catalog import Catalog
from src.retrieval.embedder import EmbedderClient
from src.retrieval.hybrid import rrf_merge
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
]

TOP_N = 5


def _fmt(rows: list[dict]) -> str:
    if not rows:
        return "  (no results)"
    return "\n".join(
        f"  {i+1}. [{p['category'][:14]:14}] {p['name'][:90]}" for i, p in enumerate(rows)
    )


async def main() -> None:
    logging.basicConfig(level=logging.WARNING)
    vs = VectorStore.load(settings.indices_dir)
    bm25 = BM25Store.load(settings.indices_dir)
    catalog = Catalog.load(settings.indices_dir)
    embedder = EmbedderClient()

    try:
        for q in QUERIES:
            print("=" * 100)
            print(f"QUERY: {q!r}")
            print("=" * 100)

            sparse_hits = bm25.search(tokenize(q), settings.top_k_sparse)
            bm25_rows = catalog.many([row for row, _ in sparse_hits[:TOP_N]])
            print(f"\n[BM25-only top-{TOP_N}]")
            print(_fmt(bm25_rows))

            vec = await embedder.embed(q)
            dense_hits = vs.search(vec, settings.top_k_dense)
            dense_rows = catalog.many([row for row, _ in dense_hits[:TOP_N]])
            print(f"\n[Dense-only top-{TOP_N}]")
            print(_fmt(dense_rows))

            merged_rows = rrf_merge(dense_hits, sparse_hits, k=settings.rrf_k, top=TOP_N)
            hybrid_rows = catalog.many(merged_rows)
            print(f"\n[Hybrid (BM25+dense, RRF) top-{TOP_N}]")
            print(_fmt(hybrid_rows))
            print()
    finally:
        await embedder.aclose()


if __name__ == "__main__":
    asyncio.run(main())
