from __future__ import annotations

import asyncio
import logging
from typing import Any

import httpx

from src.config import settings

log = logging.getLogger(__name__)

# Catch the whole TransportError family (connect, read, write, protocol,
# pool, timeout variants — all subclasses). The previous narrow list missed
# ReadTimeout, which is what fires when llama.cpp's thinking-mode response
# takes longer than http_timeout.
_RETRYABLE_EXCEPTIONS = (
    httpx.TransportError,
    httpx.RemoteProtocolError,
)


class LLMClient:
    """Async client for the llama.cpp HTTP server running LFM2.5."""

    def __init__(self, base_url: str | None = None, timeout: float | None = None) -> None:
        self._base = (base_url or settings.llm_url).rstrip("/")
        # Disable keep-alive: long thinking-mode LLM calls cause the server to
        # drop idle connections, and a stale pooled connection then yields a
        # ConnectError on the next request. Fresh connection per call is more
        # reliable than chasing keep-alive timeouts.
        self._client = httpx.AsyncClient(
            timeout=timeout or settings.http_timeout,
            limits=httpx.Limits(max_keepalive_connections=0, max_connections=8),
        )

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

        last_exc: Exception | None = None
        for attempt in range(3):
            try:
                r = await self._client.post(f"{self._base}/v1/chat/completions", json=payload)
                r.raise_for_status()
                data = r.json()
                try:
                    return data["choices"][0]["message"]["content"].strip()
                except (KeyError, IndexError, AttributeError) as exc:
                    log.error("Unexpected LLM response shape: %s", data)
                    raise RuntimeError("malformed LLM response") from exc
            except _RETRYABLE_EXCEPTIONS as exc:
                last_exc = exc
                wait = 0.5 * (2 ** attempt)  # 0.5, 1.0, 2.0
                log.warning(
                    "LLM call attempt %d failed (%s), retrying in %.1fs",
                    attempt + 1, type(exc).__name__, wait,
                )
                await asyncio.sleep(wait)
        assert last_exc is not None
        raise last_exc

    async def health(self) -> bool:
        try:
            r = await self._client.get(f"{self._base}/health", timeout=5.0)
            return r.status_code == 200
        except httpx.HTTPError:
            return False
