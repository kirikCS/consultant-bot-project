from __future__ import annotations

import argparse
import asyncio

from src.config import settings
from src.retrieval.bm25_store import BM25Store
from src.retrieval.catalog import Catalog
from src.retrieval.embedder import EmbedderClient
from src.retrieval.hybrid import rrf_merge
from src.retrieval.tokenize import tokenize
from src.retrieval.vector_store import VectorStore


async def run(query: str) -> None:
    embedder = EmbedderClient()
    vs = VectorStore.load(settings.indices_dir)
    bm25 = BM25Store.load(settings.indices_dir)
    catalog = Catalog.load(settings.indices_dir)

    print(f"\nQuery: {query!r}\n")

    vec = await embedder.embed(query)
    dense = vs.search(vec, settings.top_k_dense)
    sparse = bm25.search(tokenize(query), settings.top_k_sparse)
    merged = rrf_merge(dense, sparse, k=settings.rrf_k, top=settings.rerank_input)

    print("--- Dense top-10 ---")
    for row, score in dense[:10]:
        p = catalog.get(row)
        print(f"  {score:+.3f}  {p['name']}  [{p['category']}]")

    print("\n--- Sparse (BM25) top-10 ---")
    for row, score in sparse[:10]:
        p = catalog.get(row)
        print(f"  {score:+.3f}  {p['name']}  [{p['category']}]")

    print("\n--- RRF merged top-10 ---")
    for row in merged:
        p = catalog.get(row)
        print(f"  {p['name']}  [{p['category']}]  price={p['price']}")

    await embedder.aclose()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("query")
    args = parser.parse_args()
    asyncio.run(run(args.query))


if __name__ == "__main__":
    main()
