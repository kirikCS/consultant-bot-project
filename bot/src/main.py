"""Точка входа бота: ждёт upstream-сервисы, собирает индексы при необходимости, поднимает long-polling Telegram."""
from __future__ import annotations

import asyncio
import logging

from src.agent.loop import Agent
from src.bot import build_bot, build_dispatcher
from src.config import settings
from src.indexer.build_index import artifacts_exist, build as build_index
from src.llm.client import LLMClient
from src.memory.short_term import ShortTermMemory
from src.pipeline import Pipeline
from src.retrieval.bm25_store import BM25Store
from src.retrieval.catalog import Catalog
from src.retrieval.embedder import EmbedderClient
from src.retrieval.tfidf_store import TfidfStore
from src.retrieval.vector_store import VectorStore
from src.tools.registry import ToolRegistry
from src.tools.search_memory import SearchMemoryTool
from src.tools.search_services import SearchServicesTool

log = logging.getLogger(__name__)


async def _wait_for_services(llm: LLMClient, emb: EmbedderClient, attempts: int = 60) -> None:
    for i in range(attempts):
        ok_llm = await llm.health()
        ok_emb = await emb.health()
        if ok_llm and ok_emb:
            log.info("upstream services healthy after %d attempts", i + 1)
            return
        log.info("waiting for upstream (llm=%s emb=%s) attempt=%d", ok_llm, ok_emb, i + 1)
        await asyncio.sleep(2.0)
    raise RuntimeError("upstream LLM/embedder did not become healthy in time")


async def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    if settings.telegram_bot_token in ("", "changeme"):
        raise RuntimeError("TELEGRAM_BOT_TOKEN is not set in environment")

    llm = LLMClient()
    embedder = EmbedderClient()

    await _wait_for_services(llm, embedder)

    if not artifacts_exist():
        log.info("Building retrieval indices (first-time setup)…")
        await build_index()
    else:
        log.info("Retrieval indices already present")

    vs = VectorStore.load(settings.indices_dir)
    bm25 = BM25Store.load(settings.indices_dir)
    tfidf = TfidfStore.load(settings.indices_dir)
    catalog = Catalog.load(settings.indices_dir)

    memory = ShortTermMemory()
    await memory.connect()

    tools = ToolRegistry(
        search_services=SearchServicesTool(
            vector_store=vs,
            bm25_store=bm25,
            tfidf_store=tfidf,
            catalog=catalog,
            embedder=embedder,
        ),
        search_memory=SearchMemoryTool(memory=memory),
    )
    agent = Agent(llm=llm, tools=tools)

    pipeline = Pipeline(agent=agent, tools=tools, memory=memory)

    bot = build_bot(settings.telegram_bot_token)
    dp = build_dispatcher(pipeline)

    try:
        log.info("Bot starting (long polling)")
        await dp.start_polling(bot, allowed_updates=dp.resolve_used_update_types())
    finally:
        log.info("Shutting down…")
        await bot.session.close()
        await memory.close()
        await llm.aclose()
        await embedder.aclose()


if __name__ == "__main__":
    asyncio.run(main())
