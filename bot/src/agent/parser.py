"""Parse tool-call JSON from LLM output.

Ported from localscript-agent/llm.py — multi-strategy parser robust to
formatting noise (code fences, mixed prose, balanced-brace recovery).
"""
from __future__ import annotations

import json
import re
from typing import Any

_THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL)
_TRAILING_THINK = re.compile(r"<think>.*$", re.DOTALL)


def strip_thinking(text: str) -> str:
    """Remove <think>...</think> blocks (closed or truncated)."""
    text = _THINK_RE.sub("", text)
    text = _TRAILING_THINK.sub("", text)
    return text.strip()


def _try_parse(s: str) -> Any:
    try:
        return json.loads(s)
    except json.JSONDecodeError:
        return None


def _find_balanced_json(text: str) -> Any:
    """Find the first {...} block with balanced braces, return parsed dict or None."""
    start = text.find("{")
    while start != -1:
        depth = 0
        in_str = False
        esc = False
        for i in range(start, len(text)):
            ch = text[i]
            if in_str:
                if esc:
                    esc = False
                elif ch == "\\":
                    esc = True
                elif ch == '"':
                    in_str = False
            else:
                if ch == '"':
                    in_str = True
                elif ch == "{":
                    depth += 1
                elif ch == "}":
                    depth -= 1
                    if depth == 0:
                        candidate = text[start : i + 1]
                        parsed = _try_parse(candidate)
                        if isinstance(parsed, dict):
                            return parsed
                        break
        start = text.find("{", start + 1)
    return None


def _normalize(d: dict) -> dict:
    """Normalize common field-name variants the LLM may emit."""
    if "tool" not in d:
        for alt in ("name", "function", "tool_name"):
            if alt in d:
                d["tool"] = d.pop(alt)
                break
    if isinstance(d.get("arguments"), dict):
        # OpenAI-style: {"tool": "...", "arguments": {...}}
        args = d.pop("arguments")
        for k, v in args.items():
            d.setdefault(k, v)
    return d


def parse_tool_call(text: str, known_tools: set[str]) -> dict | None:
    """Try to extract a JSON tool call from text. Return dict or None.

    Multi-strategy parse: direct JSON → ```json fence → balanced-brace recovery.
    Only returns a dict whose 'tool' field names a known tool.
    """
    text = strip_thinking(text)
    if not text:
        return None

    # 1) Direct JSON parse
    r = _try_parse(text)
    if isinstance(r, dict):
        r = _normalize(r)
        if r.get("tool") in known_tools:
            return r

    # 2) JSON inside ```json fence
    m = re.search(r"```json\s*(\{.*?\})\s*```", text, re.DOTALL)
    if m:
        r = _try_parse(m.group(1))
        if isinstance(r, dict):
            r = _normalize(r)
            if r.get("tool") in known_tools:
                return r

    # 3) Any fenced code block containing JSON
    m = re.search(r"```\s*(\{.*?\})\s*```", text, re.DOTALL)
    if m:
        r = _try_parse(m.group(1))
        if isinstance(r, dict):
            r = _normalize(r)
            if r.get("tool") in known_tools:
                return r

    # 4) Balanced-brace recovery (handles trailing prose, leading prose)
    r = _find_balanced_json(text)
    if isinstance(r, dict):
        r = _normalize(r)
        if r.get("tool") in known_tools:
            return r

    return None
