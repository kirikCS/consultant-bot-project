"""Multi-turn CLI probe that exercises the full agentic pipeline.

Usage:
  python -m src.retrieval.cli "<turn 1>" "<turn 2>" ...
"""
from __future__ import annotations

import argparse
import asyncio
import logging

from src.agent.loop import Agent
from src.config import settings
from src.llm.client import LLMClient
from src.memory.short_term import ShortTermMemory
from src.pipeline import Pipeline
from src.retrieval.bm25_store import BM25Store
from src.retrieval.catalog import Catalog
from src.retrieval.embedder import EmbedderClient
from src.retrieval.vector_store import VectorStore
from src.tools.registry import ToolRegistry
from src.tools.search_memory import SearchMemoryTool
from src.tools.search_services import SearchServicesTool


async def run(queries: list[str]) -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    llm = LLMClient()
    embedder = EmbedderClient()
    vs = VectorStore.load(settings.indices_dir)
    bm25 = BM25Store.load(settings.indices_dir)
    catalog = Catalog.load(settings.indices_dir)

    memory = ShortTermMemory(db_path=":memory:")
    await memory.connect()

    tools = ToolRegistry(
        search_services=SearchServicesTool(
            vector_store=vs, bm25_store=bm25, catalog=catalog, embedder=embedder
        ),
        search_memory=SearchMemoryTool(memory=memory),
    )
    agent = Agent(llm=llm, tools=tools)
    pipeline = Pipeline(agent=agent, tools=tools, memory=memory)

    chat_id = 0
    for i, q in enumerate(queries, 1):
        print(f"\n=== Turn {i} — Клиент: {q!r} ===")
        answer = await pipeline.handle(chat_id=chat_id, user_text=q)
        print(f"--- Анна:\n{answer}")

    await memory.close()
    await llm.aclose()
    await embedder.aclose()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Drive the bot through a sequence of user turns."
    )
    parser.add_argument("queries", nargs="+", help="One or more user turns")
    args = parser.parse_args()
    asyncio.run(run(args.queries))


if __name__ == "__main__":
    main()
