"""Tool: search the medical-services catalog (FAISS + BM25 + RRF)."""
from __future__ import annotations

import logging

from src.config import settings
from src.retrieval.bm25_store import BM25Store
from src.retrieval.catalog import Catalog
from src.retrieval.embedder import EmbedderClient
from src.retrieval.hybrid import rrf_merge
from src.retrieval.tokenize import tokenize
from src.retrieval.vector_store import VectorStore

log = logging.getLogger(__name__)


class SearchServicesTool:
    name = "search_services"

    def __init__(
        self,
        *,
        vector_store: VectorStore,
        bm25_store: BM25Store,
        catalog: Catalog,
        embedder: EmbedderClient,
    ) -> None:
        self._vs = vector_store
        self._bm25 = bm25_store
        self._catalog = catalog
        self._embedder = embedder

    async def run(self, query: str, top_k: int = 5) -> str:
        if not query or not query.strip():
            return "Ошибка: пустой запрос. Передай поле 'query' с поисковой фразой."

        top_k = max(1, min(int(top_k or 5), 10))

        vec = await self._embedder.embed(query)
        dense_hits = self._vs.search(vec, settings.top_k_dense)
        sparse_hits = self._bm25.search(tokenize(query), settings.top_k_sparse)
        rows = rrf_merge(dense_hits, sparse_hits, k=settings.rrf_k, top=top_k)
        items = self._catalog.many(rows)

        if not items:
            return f"По запросу «{query}» ничего не найдено в каталоге."

        lines = [f"По запросу «{query}» найдено {len(items)} услуг(и):"]
        for i, p in enumerate(items, 1):
            lines.append(
                f"{i}. {p['name']} — {p['price']} руб. "
                f"(категория: {p['category']}, группа: {p['group']}, id: {p['id']})"
            )
        return "\n".join(lines)
