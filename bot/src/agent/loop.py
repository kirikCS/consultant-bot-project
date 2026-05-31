"""Agent loop: LLM-вызов -> strip thinking -> парс tool-call -> диспатч инструмента -> инжект результата как user-сообщение"""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field

from src.agent.parser import parse_tool_call, strip_thinking
from src.config import settings
from src.llm.client import LLMClient
from src.tools.registry import ToolRegistry

log = logging.getLogger(__name__)

_CATALOG_SIGNALS = re.compile(
    r"(?:"
    r"\bцен[аы]\b|\bстоит\b|\bстоимост|"
    r"\bесть\s+ли\b|\bимеется\b|\bимеете\b|\bу\s+вас\s+есть\b|\bесть\s+в\s+каталог|"
    r"\bкак(ие|ой|ое|ую|ая)\b.{0,40}\b(услуг|анализ|процедур|вариант|абонемент|тариф|препарат)|"
    r"\bпро\s+\w+|"
    r"\bImmunoHealth|\bImmunoCap|\bImmunoCAP|"
    r"\bи?мм?уно[хh]?[еэ]лс|\bи?мм?уно[кc]ап|"
    r"\bкаталог\w*|"
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

        last_user_text = next(
            (m["content"] for m in reversed(messages) if m.get("role") == "user"),
            "",
        )

        prev_assistant_was_question = False
        for m in reversed(messages[:-1]):
            if m.get("role") == "assistant":
                c = (m.get("content") or "").rstrip().rstrip('"”»\').*')
                prev_assistant_was_question = "?" in c[-15:]
                break

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
                ends_with_question = cleaned.rstrip().endswith(("?", "?!"))
                trigger_catalog = (
                    _looks_like_catalog_question(last_user_text)
                    and not ends_with_question
                )
                trigger_post_clarify = prev_assistant_was_question
                should_force_search = (
                    i == 0
                    and not tool_calls
                    and (trigger_catalog or trigger_post_clarify)
                )
                if should_force_search:
                    log.info(
                        "safety-net: forcing search_services (catalog=%s clarify=%s)",
                        trigger_catalog, trigger_post_clarify,
                    )
                    user_turns = [
                        m["content"] for m in messages if m.get("role") == "user"
                    ]
                    fallback_query = (
                        " ".join(user_turns[-2:]) if len(user_turns) >= 2 else last_user_text
                    )
                    fallback_call = {
                        "tool": "search_services",
                        "query": fallback_query,
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

                return AgentRun(
                    final_text=cleaned or raw.strip(),
                    tool_calls=tool_calls,
                    iterations=i + 1,
                )

            tool_calls.append(call)
            msgs.append({"role": "assistant", "content": cleaned})

            result = await self._tools.dispatch(chat_id=chat_id, call=call)
            log.info(
                "agent it=%d tool=%s ok=%s result_len=%d",
                i + 1, result.name, result.ok, len(result.content),
            )
            msgs.append({"role": "user", "content": result.content})

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
