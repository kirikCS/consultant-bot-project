"""Tool: search older conversation history (BM25 over SQLite-backed turns)."""
from __future__ import annotations

import logging

from src.memory.short_term import ShortTermMemory

log = logging.getLogger(__name__)


class SearchMemoryTool:
    name = "search_memory"

    def __init__(self, memory: ShortTermMemory) -> None:
        self._memory = memory

    async def run(self, chat_id: int, query: str, top_k: int = 3) -> str:
        if not query or not query.strip():
            return "Ошибка: пустой запрос. Передай 'query' с фразой для поиска."

        top_k = max(1, min(int(top_k or 3), 8))

        # Pull from older history (skip the pinned window since the LLM already sees that)
        turns = await self._memory.recall_excluding_pinned(
            chat_id, query, top_k=top_k
        )

        if not turns:
            return f"По запросу «{query}» в истории диалога ничего не найдено."

        lines = [f"Найдено {len(turns)} релевантных фрагмент(ов) из истории:"]
        for i, t in enumerate(turns, 1):
            role = "клиент" if t["role"] == "user" else "ассистент"
            body = t["content"]
            if len(body) > 300:
                body = body[:300] + "…"
            lines.append(f"--- Фрагмент {i} ({role}) ---\n{body}")
        return "\n".join(lines)
