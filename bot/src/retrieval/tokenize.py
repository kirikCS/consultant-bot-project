"""Простая Unicode-токенизация для BM25-индексов и query-side обработки."""
from __future__ import annotations

import re

_TOKEN_RE = re.compile(r"[\w]+", flags=re.UNICODE)


def tokenize(text: str) -> list[str]:
    return [t.lower() for t in _TOKEN_RE.findall(text or "")]
