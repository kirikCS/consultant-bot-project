from __future__ import annotations

import logging
import re

from src.config import settings
from src.llm.client import LLMClient
from src.llm.prompts import RERANK_PROMPT, format_numbered

log = logging.getLogger(__name__)

_NUM_RE = re.compile(r"\d+")


class Reranker:
    """LFM self-rerank: ask the LLM to pick top-N candidates by index."""

    def __init__(self, llm: LLMClient) -> None:
        self._llm = llm

    async def rerank(self, query: str, candidates: list[dict], top_n: int) -> list[dict]:
        if not candidates:
            return []
        if len(candidates) <= top_n:
            return candidates

        prompt = RERANK_PROMPT.format(
            query=query,
            numbered_candidates=format_numbered(candidates),
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
        # Pad with RRF order if the LLM returned fewer unique indices than asked
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
