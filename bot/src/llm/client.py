from __future__ import annotations

import logging
from typing import Any

import httpx

from src.config import settings

log = logging.getLogger(__name__)


class LLMClient:
    """Async client for the llama.cpp HTTP server running LFM2.5."""

    def __init__(self, base_url: str | None = None, timeout: float | None = None) -> None:
        self._base = (base_url or settings.llm_url).rstrip("/")
        self._client = httpx.AsyncClient(timeout=timeout or settings.http_timeout)

    async def aclose(self) -> None:
        await self._client.aclose()

    async def chat(
        self,
        messages: list[dict[str, str]],
        *,
        temperature: float,
        max_tokens: int,
        stop: list[str] | None = None,
        thinking: bool = True,
    ) -> str:
        payload: dict[str, Any] = {
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "stream": False,
            "chat_template_kwargs": {"enable_thinking": thinking},
        }
        if stop:
            payload["stop"] = stop

        r = await self._client.post(f"{self._base}/v1/chat/completions", json=payload)
        r.raise_for_status()
        data = r.json()
        try:
            return data["choices"][0]["message"]["content"].strip()
        except (KeyError, IndexError, AttributeError) as exc:
            log.error("Unexpected LLM response shape: %s", data)
            raise RuntimeError("malformed LLM response") from exc

    async def health(self) -> bool:
        try:
            r = await self._client.get(f"{self._base}/health", timeout=5.0)
            return r.status_code == 200
        except httpx.HTTPError:
            return False
