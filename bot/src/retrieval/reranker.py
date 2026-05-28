"""LFM self-rerank (опционально): просим саму LLM выбрать top-N из набора кандидатов RRF."""
from __future__ import annotations

import logging
import re

from src.config import settings
from src.llm.client import LLMClient

log = logging.getLogger(__name__)

_NUM_RE = re.compile(r"\d+")

_RERANK_PROMPT = """Из списка ниже выбери {top_n} услуг, наиболее точно соответствующие вопросу клиента.

Вопрос: {query}

Услуги:
{numbered_candidates}

Ответь ТОЛЬКО номерами через запятую, без пояснений. Пример: 4, 1, 7"""


def _format_numbered(payloads: list[dict]) -> str:
    return "\n".join(
        f"{i + 1}. {p['name']} (категория: {p['category']}, цена: {p['price']} руб.)"
        for i, p in enumerate(payloads)
    )


class Reranker:
    def __init__(self, llm: LLMClient) -> None:
        self._llm = llm

    async def rerank(self, query: str, candidates: list[dict], top_n: int) -> list[dict]:
        if not candidates:
            return []
        if len(candidates) <= top_n:
            return candidates

        prompt = _RERANK_PROMPT.format(
            query=query,
            top_n=top_n,
            numbered_candidates=_format_numbered(candidates),
        )
        try:
            text = await self._llm.chat(
                [{"role": "user", "content": prompt}],
                temperature=settings.rerank_temperature,
                max_tokens=settings.rerank_max_tokens,
                thinking=False,
            )
        except Exception as exc:
            log.warning("rerank LLM call failed (%s); falling back to RRF order", exc)
            return candidates[:top_n]

        picked = self._parse(text, n_candidates=len(candidates), top_n=top_n)
        if not picked:
            log.warning("rerank parse failed for output %r; using RRF order", text)
            return candidates[:top_n]

        result = [candidates[i] for i in picked]
        if len(result) < top_n:
            for c in candidates:
                if c not in result:
                    result.append(c)
                if len(result) == top_n:
                    break
        return result

    @staticmethod
    def _parse(text: str, *, n_candidates: int, top_n: int) -> list[int]:
        seen: list[int] = []
        for m in _NUM_RE.finditer(text):
            n = int(m.group())
            if 1 <= n <= n_candidates and (n - 1) not in seen:
                seen.append(n - 1)
            if len(seen) == top_n:
                break
        return seen
