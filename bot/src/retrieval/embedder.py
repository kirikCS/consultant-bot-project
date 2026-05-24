from __future__ import annotations

import asyncio
import logging

import httpx
import numpy as np

from src.config import settings

log = logging.getLogger(__name__)


class EmbedderClient:
    """Async client for the llama.cpp embedding HTTP server (nomic-embed-text)."""

    def __init__(self, base_url: str | None = None, timeout: float | None = None) -> None:
        self._base = (base_url or settings.embedder_url).rstrip("/")
        self._client = httpx.AsyncClient(timeout=timeout or settings.http_timeout)

    async def aclose(self) -> None:
        await self._client.aclose()

    async def embed(self, text: str) -> np.ndarray:
        vec = (await self.embed_batch([text]))[0]
        return vec

    async def embed_batch(self, texts: list[str]) -> np.ndarray:
        """Embed each text concurrently — llama.cpp /embedding takes one input at a time."""
        results = await asyncio.gather(*(self._embed_one(t) for t in texts))
        arr = np.asarray(results, dtype=np.float32)
        norms = np.linalg.norm(arr, axis=1, keepdims=True)
        norms[norms == 0.0] = 1.0
        return arr / norms

    async def _embed_one(self, text: str) -> list[float]:
        # llama.cpp accepts {"content": "..."} on /embedding
        r = await self._client.post(f"{self._base}/embedding", json={"content": text})
        r.raise_for_status()
        data = r.json()

        # Response shape can be {"embedding": [...]} or [{"embedding": [...]}]
        if isinstance(data, list):
            data = data[0]
        emb = data.get("embedding")
        if emb is None:
            raise RuntimeError(f"no 'embedding' field in response: {data}")
        # Sometimes returned as [[...]]
        if isinstance(emb, list) and emb and isinstance(emb[0], list):
            emb = emb[0]
        return emb

    async def health(self) -> bool:
        try:
            r = await self._client.get(f"{self._base}/health", timeout=5.0)
            return r.status_code == 200
        except httpx.HTTPError:
            return False
