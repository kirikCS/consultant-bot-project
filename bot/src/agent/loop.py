"""Agent loop: LLM → parse → tool dispatch → result-injection, bounded by max_iters.

Mirrors the FSM from localscript-agent/agent.py but trimmed to two states
that matter for a Q&A bot:

  LLM_INFERENCE   — call the model, append assistant turn (with <think> stripped)
  PARSE_AND_ROUTE — if output is a tool call, dispatch + append result, loop;
                    otherwise return the text as the final answer.
"""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field

from src.agent.parser import parse_tool_call, strip_thinking
from src.config import settings
from src.llm.client import LLMClient
from src.tools.registry import ToolRegistry

log = logging.getLogger(__name__)

# Lightweight intent signal: words/patterns indicating the turn is about a
# catalog item, not just chit-chat. If present in the LAST user turn AND the
# model's first move is plain text (no tool call) AND no search has been done
# yet this turn, we insert ONE search_services call as a safety net. This is
# NOT a forced pipeline step — it only kicks in when the model fails to
# recognize a catalog question.
_CATALOG_SIGNALS = re.compile(
    r"(?:"
    r"\bцен[аы]\b|\bстоит\b|\bстоимост|"
    r"\bесть\s+ли\b|\bимеется\b|\bимеете\b|\bу\s+вас\s+есть\b|"
    r"\bкак(ие|ой|ое|ую|ая)\b.{0,40}\b(услуг|анализ|процедур|вариант|абонемент|тариф|препарат)|"
    r"\bпро\s+\w+|"
    r"\bImmunoHealth|\bImmunoCap|\bImmunoCAP|"
    r"\bанализ\b|\bпроцедур|\bабонемент|\bуслуг"
    r")",
    re.IGNORECASE,
)


def _looks_like_catalog_question(text: str) -> bool:
    return bool(_CATALOG_SIGNALS.search(text or ""))


@dataclass
class AgentRun:
    final_text: str
    tool_calls: list[dict] = field(default_factory=list)
    iterations: int = 0


class Agent:
    def __init__(self, llm: LLMClient, tools: ToolRegistry) -> None:
        self._llm = llm
        self._tools = tools

    async def run(
        self,
        *,
        chat_id: int,
        messages: list[dict],
        max_iters: int | None = None,
    ) -> AgentRun:
        max_iters = max_iters if max_iters is not None else settings.agent_max_iters
        msgs = list(messages)
        tool_calls: list[dict] = []

        # Last raw user turn (the catalog-question detector keys off this)
        last_user_text = next(
            (m["content"] for m in reversed(messages) if m.get("role") == "user"),
            "",
        )

        for i in range(max_iters):
            raw = await self._llm.chat(
                msgs,
                temperature=settings.answer_temperature,
                max_tokens=settings.answer_max_tokens,
            )
            cleaned = strip_thinking(raw)
            log.info("agent it=%d raw_len=%d clean_len=%d", i + 1, len(raw), len(cleaned))

            call = parse_tool_call(raw, self._tools.known_tools)
            if call is None:
                # Safety net: if the FIRST move is plain prose but the user
                # turn was clearly a catalog question, inject one search and
                # let the model answer on the next iteration. This keeps the
                # tool-as-tool design but covers small-model misses.
                if (
                    i == 0
                    and not tool_calls
                    and _looks_like_catalog_question(last_user_text)
                ):
                    log.info("safety-net: forcing search_services for catalog-like input")
                    fallback_call = {
                        "tool": "search_services",
                        "query": last_user_text,
                        "top_k": 5,
                    }
                    tool_calls.append(fallback_call)
                    msgs.append({"role": "assistant", "content": cleaned})
                    result = await self._tools.dispatch(chat_id=chat_id, call=fallback_call)
                    msgs.append(
                        {
                            "role": "user",
                            "content": (
                                f"[Системная подсказка] Я выполнила за тебя поиск по каталогу. "
                                f"Используй эти результаты для ответа клиенту обычным текстом:\n\n{result.content}"
                            ),
                        }
                    )
                    continue

                # Plain prose → final answer
                return AgentRun(
                    final_text=cleaned or raw.strip(),
                    tool_calls=tool_calls,
                    iterations=i + 1,
                )

            tool_calls.append(call)
            # Append assistant's tool call to context so the model sees what it asked for
            msgs.append({"role": "assistant", "content": cleaned})

            # Execute tool, inject result as user-role message
            result = await self._tools.dispatch(chat_id=chat_id, call=call)
            log.info(
                "agent it=%d tool=%s ok=%s result_len=%d",
                i + 1, result.name, result.ok, len(result.content),
            )
            msgs.append({"role": "user", "content": result.content})

        # Hit the ceiling — force the model to produce a final answer using gathered context
        msgs.append(
            {
                "role": "user",
                "content": (
                    "Достаточно поисков. Дай клиенту краткий человеческий ответ "
                    "обычным текстом (без JSON), используя собранную выше информацию."
                ),
            }
        )
        raw = await self._llm.chat(
            msgs,
            temperature=settings.answer_temperature,
            max_tokens=settings.answer_max_tokens,
        )
        final = strip_thinking(raw) or raw.strip()
        return AgentRun(final_text=final, tool_calls=tool_calls, iterations=max_iters + 1)
