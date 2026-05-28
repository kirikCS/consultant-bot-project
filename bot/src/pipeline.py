"""Агентный пайплайн обработки одного хода: input filter → сборка контекста → агент-цикл → output filter."""
from __future__ import annotations

import logging
import time

from src.agent.loop import Agent
from src.config import settings
from src.filters.input_filter import filter_input
from src.filters.output_filter import filter_output
from src.llm.prompts import GREETING, SYSTEM_PROMPT
from src.memory.short_term import ShortTermMemory
from src.tools.registry import ToolRegistry

log = logging.getLogger(__name__)


class Pipeline:
    def __init__(
        self,
        *,
        agent: Agent,
        tools: ToolRegistry,
        memory: ShortTermMemory,
    ) -> None:
        self._agent = agent
        self._tools = tools
        self._memory = memory

    async def is_new_chat(self, chat_id: int) -> bool:
        return await self._memory.is_new_chat(chat_id)

    async def mark_greeted(self, chat_id: int) -> None:
        await self._memory.append(chat_id, "assistant", GREETING)

    async def handle(self, chat_id: int, user_text: str) -> str:
        t0 = time.perf_counter()

        flt = filter_input(user_text)
        if not flt.passed:
            return flt.refusal or "Извините, я не понял запрос."

        cleaned = flt.cleaned

        history = await self._memory.build_context(chat_id, cleaned)
        t_ctx = time.perf_counter()

        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            *history,
            {"role": "user", "content": cleaned},
        ]

        try:
            run = await self._agent.run(chat_id=chat_id, messages=messages)
        except Exception as exc:
            log.exception("agent run failed: %s", exc)
            return (
                "Извините, я временно не могу обработать запрос. "
                "Попробуйте, пожалуйста, ещё раз через минуту."
            )

        t_agent = time.perf_counter()
        final = filter_output(run.final_text)

        await self._memory.append(chat_id, "user", cleaned)
        await self._memory.append(chat_id, "assistant", final)

        log.info(
            "chat=%s ctx=%d iters=%d tools=%s timings(ms) ctx=%.0f agent=%.0f total=%.0f",
            chat_id,
            len(history),
            run.iterations,
            [c.get("tool") for c in run.tool_calls],
            (t_ctx - t0) * 1000,
            (t_agent - t_ctx) * 1000,
            (t_agent - t0) * 1000,
        )
        return final
