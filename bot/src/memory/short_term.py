"""Per-chat conversation memory: SQLite store + rolling-window context builder.

Architecture ports the localscript-agent design:

- Every turn is appended to SQLite (durable, survives restarts).
- `build_context()` composes the LLM `messages` list as:
    [system stub omitted — pipeline adds it,
     retrieved-memory block as a synthetic user message (if older relevant turns found),
     ...pinned-recent raw turns]
- Older history (everything before the pinned tail) is searched via BM25 +
  prefix-overlap; the highest-scoring excerpts are rendered into the retrieved
  block.
- Auto-compaction fires when the char budget is exceeded: the older 70% of
  iterations is replaced with a deterministic summary (last user task + last
  assistant turn + service ids mentioned), the most recent 30% is preserved
  verbatim. No extra LLM call — cheap and deterministic.
"""
from __future__ import annotations

import re
import time
from pathlib import Path

import aiosqlite
from rank_bm25 import BM25Okapi

from src.config import settings
from src.retrieval.tokenize import tokenize

_SCHEMA = Path(__file__).with_name("schema.sql").read_text()
_SERVICE_ID_RE = re.compile(r"\b\d{11}\b")

# Canned refusal openings — both from Python filters and recurring LLM
# self-mirror copies. If a stored assistant turn STARTS WITH any of these,
# it (and the user question that triggered it) is excluded from the rolling
# context. Otherwise prior refusals teach the model to refuse new legit
# questions by pattern-matching.
_REFUSAL_PREFIXES = (
    "я не даю медицинских",
    "извините, я не даю медицинских",
    "извините, я могу отвечать только на вопросы",
    "я отвечаю только на вопросы",
    "извините, в ответе произошёл сбой",
    "извините, я не смог сформировать",
    "извините, я не понял запрос",
    "пустой запрос",
)


def _is_filter_refusal(content: str) -> bool:
    if not content:
        return False
    head = content.strip().lower()[:80]
    return head.startswith(_REFUSAL_PREFIXES)


def _drop_refusal_pairs(turns: list[dict]) -> list[dict]:
    """Remove (user-question → assistant-refusal) pairs from chronological turns.

    Prevents the small model from mirroring earlier canned refusals onto new,
    legitimate queries — exactly the contamination problem the user reported.
    """
    skip: set[int] = set()
    for i, t in enumerate(turns):
        if t.get("role") == "assistant" and _is_filter_refusal(t.get("content", "")):
            skip.add(i)
            if i > 0 and turns[i - 1].get("role") == "user":
                skip.add(i - 1)
    if not skip:
        return turns
    return [t for i, t in enumerate(turns) if i not in skip]


def _approx_tokens(text: str) -> int:
    """Rough chars/token estimate for mixed RU/EN."""
    return max(1, len(text) // 3)


def _approx_chars(turns: list[dict]) -> int:
    return sum(len(t.get("content", "")) + 8 for t in turns)


class ShortTermMemory:
    def __init__(self, db_path: str | None = None) -> None:
        self._db_path = db_path or settings.sqlite_path
        self._conn: aiosqlite.Connection | None = None

    async def connect(self) -> None:
        Path(self._db_path).parent.mkdir(parents=True, exist_ok=True)
        self._conn = await aiosqlite.connect(self._db_path)
        await self._conn.executescript(_SCHEMA)
        await self._conn.commit()

    async def close(self) -> None:
        if self._conn is not None:
            await self._conn.close()
            self._conn = None

    async def is_new_chat(self, chat_id: int) -> bool:
        assert self._conn is not None
        async with self._conn.execute(
            "SELECT 1 FROM turns WHERE chat_id = ? LIMIT 1", (chat_id,)
        ) as cur:
            return (await cur.fetchone()) is None

    async def append(self, chat_id: int, role: str, content: str) -> None:
        assert self._conn is not None
        await self._conn.execute(
            "INSERT INTO turns(chat_id, role, content, ts) VALUES (?, ?, ?, ?)",
            (chat_id, role, content, int(time.time())),
        )
        await self._conn.commit()

    async def _all_turns(self, chat_id: int) -> list[dict]:
        assert self._conn is not None
        async with self._conn.execute(
            "SELECT id, role, content, ts FROM turns "
            "WHERE chat_id = ? ORDER BY ts ASC, id ASC",
            (chat_id,),
        ) as cur:
            rows = await cur.fetchall()
        turns = [
            {"id": r[0], "role": r[1], "content": r[2], "ts": r[3]}
            for r in rows
        ]
        # Strip refusal pairs so the LLM never sees them as a pattern to copy
        return _drop_refusal_pairs(turns)

    async def recall_excluding_pinned(
        self, chat_id: int, query: str, top_k: int
    ) -> list[dict]:
        """BM25 + prefix-overlap recall over the OLDER slice (skip pinned tail).

        Returned turns are sorted chronologically.
        """
        pinned_n = max(0, int(settings.memory_pinned_recent))
        turns = await self._all_turns(chat_id)
        if not turns:
            return []
        older = turns[:-pinned_n] if pinned_n > 0 else turns
        if not older:
            return []

        q_tokens = tokenize(query)
        if not q_tokens:
            return older[-top_k:]

        tokenized = [tokenize(t["content"]) for t in older]
        if not any(tokenized):
            return older[-top_k:]
        bm25 = BM25Okapi([toks or [""] for toks in tokenized])
        scores = bm25.get_scores(q_tokens)
        max_score = max(scores) or 1.0

        scored = []
        for i, t in enumerate(older):
            prefix = _prefix_overlap(q_tokens, tokenized[i])
            score = 0.5 * (scores[i] / max_score) + 0.5 * prefix
            if score > 0.05:
                scored.append((score, t))
        scored.sort(key=lambda x: -x[0])
        picked = [t for _, t in scored[:top_k]]
        picked.sort(key=lambda t: t["ts"])
        return picked

    async def build_context(
        self,
        chat_id: int,
        query: str,
        *,
        pinned_recent: int | None = None,
        retrieved_k: int | None = None,
        max_chars: int | None = None,
    ) -> list[dict]:
        """Return chat-message list ready to splice between system and current user.

        Shape: [<retrieved-memory block as user msg>?, ...pinned recent raw turns]
        """
        pinned_recent = pinned_recent if pinned_recent is not None else settings.memory_pinned_recent
        retrieved_k = retrieved_k if retrieved_k is not None else settings.short_term_k
        max_chars = max_chars if max_chars is not None else settings.memory_max_chars

        turns = await self._all_turns(chat_id)
        if not turns:
            return []

        pinned = turns[-pinned_recent:] if pinned_recent > 0 else []
        older = turns[: len(turns) - len(pinned)]

        # If older history is long, compact it into a single summary block
        summary_msg: dict | None = None
        if _approx_chars(older) > max_chars and len(older) >= 4:
            summary_text = _summarize_older(older)
            summary_msg = {
                "role": "user",
                "content": f"[Сводка предыдущего разговора — {len(older)} сообщений]\n{summary_text}",
            }
            older = []  # already represented by summary

        # Surface relevant excerpts from any non-summarized older slice
        retrieved_block: dict | None = None
        if older and retrieved_k > 0:
            q_tokens = tokenize(query)
            if q_tokens:
                tokenized = [tokenize(t["content"]) for t in older]
                if any(tokenized):
                    bm25 = BM25Okapi([toks or [""] for toks in tokenized])
                    scores = bm25.get_scores(q_tokens)
                    max_score = max(scores) or 1.0
                    scored = []
                    for i, t in enumerate(older):
                        prefix = _prefix_overlap(q_tokens, tokenized[i])
                        score = 0.5 * (scores[i] / max_score) + 0.5 * prefix
                        if score > 0.1:
                            scored.append((score, t))
                    scored.sort(key=lambda x: -x[0])
                    picks = [t for _, t in scored[:retrieved_k]]
                    if picks:
                        retrieved_block = {
                            "role": "user",
                            "content": _format_retrieved(picks),
                        }

        out: list[dict] = []
        if summary_msg:
            out.append(summary_msg)
        if retrieved_block:
            out.append(retrieved_block)
        out.extend({"role": t["role"], "content": t["content"]} for t in pinned)

        # Auto-compact safety net: trim from the oldest pinned end if still over budget
        while _approx_chars(out) > max_chars and len(out) > 2:
            # Always keep the most recent two turns (last user + last assistant or just last)
            del out[0]
        return out


def _prefix_overlap(a: list[str], b: list[str]) -> float:
    if not a or not b:
        return 0.0
    n = min(len(a), len(b))
    shared = 0
    for i in range(n):
        if a[i] == b[i]:
            shared += 1
        else:
            break
    return shared / n


def _format_retrieved(picks: list[dict]) -> str:
    lines = [
        f"[Подходящие фрагменты из ранней истории диалога — {len(picks)} шт.]"
    ]
    for i, t in enumerate(picks, 1):
        role = "клиент" if t["role"] == "user" else "ассистент"
        body = t["content"]
        if len(body) > 400:
            body = body[:400] + "…"
        lines.append(f"--- Фрагмент {i} ({role}) ---\n{body}")
    return "\n".join(lines)


def _summarize_older(older: list[dict]) -> str:
    """Deterministic summary: last user request + last assistant turn + service ids touched.

    No LLM call (small model unreliable at summarization). Cheap and stable.
    """
    last_user = ""
    last_assistant = ""
    for t in reversed(older):
        if not last_assistant and t["role"] == "assistant":
            last_assistant = t["content"]
        if not last_user and t["role"] == "user":
            last_user = t["content"]
        if last_user and last_assistant:
            break

    # Collect mentioned service IDs across the older slice (11-digit catalog ids)
    ids: list[str] = []
    seen: set[str] = set()
    for t in older:
        for m in _SERVICE_ID_RE.findall(t.get("content", "")):
            if m not in seen:
                seen.add(m)
                ids.append(m)

    parts: list[str] = []
    if last_user:
        snippet = last_user[:400] + ("…" if len(last_user) > 400 else "")
        parts.append(f"Последний вопрос клиента: {snippet}")
    if last_assistant:
        snippet = last_assistant[:400] + ("…" if len(last_assistant) > 400 else "")
        parts.append(f"Последний ответ ассистента: {snippet}")
    if ids:
        parts.append(f"Услуги, упомянутые в разговоре (id): {', '.join(ids[:20])}")
    parts.append("Полная история свёрнута для экономии контекста.")
    return "\n".join(parts)
