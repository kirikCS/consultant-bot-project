"""Фильтр вывода LLM: блокирует рекомендации медикаментов, утечку шаблона промпта, эхо-инъекции и не русский текст."""
from __future__ import annotations

import re

_MAX_LEN = 4000

_MED_VERBS = (
    r"(?:"
    r"пи[лть]\w*"
    r"|пь[еёюя]\w*"
    r"|пей(?:те)?"
    r"|вы?пи[лть]\w*"
    r"|вы?пь[еёюя]\w*"
    r"|вы?пей(?:те)?"
    r"|попи[лть]\w*"
    r"|пропи[лть]\w*"
    r"|приня[лт]\w*"
    r"|принима[йлтю]\w*"
    r"|принима\w*"
    r"|купи\w+\s+лекарств\w+"
    r"|лечи(?:ть|ться|тесь)\w*"
    r")"
)

_RECOMMENDATION_PATTERNS = [
    re.compile(r"\bрекоменду[ею]\s+(вам\s+|тебе\s+)?" + _MED_VERBS, re.I),
    re.compile(r"\bя\s+(вам\s+|тебе\s+)?(рекоменду[ею]|посоветую|порекомендую)\s+.{0,20}" + _MED_VERBS, re.I),
    re.compile(r"\b(вам|тебе)\s+(стоит|лучше|следует|нужно)\s+" + _MED_VERBS, re.I),
    re.compile(r"\bя\s+бы\s+(вам\s+|тебе\s+)?(посоветовал[аи]?|порекомендовал[аи]?)\s+.{0,20}" + _MED_VERBS, re.I),
    re.compile(r"\bна\s+ваш(ем)?\s+месте\s+.{0,30}" + _MED_VERBS, re.I),
    re.compile(r"\b(принимайте|пейте|выпейте)\s+\w+\s+таблет\w*", re.I),
]

_TEMPLATE_LEAKAGE = [
    re.compile(r"(?:^|[\n.])\s*каталог\s*:", re.I),
    re.compile(r"(?:^|[\n.])\s*клиент\s*:", re.I),
    re.compile(r"(?:^|[\n.])\s*вопрос\s*:", re.I),
    re.compile(r"(?:^|[\n.])\s*ответ\s*:", re.I),
    re.compile(r"(?:^|[\n.])\s*system\s*:", re.I),
    re.compile(r"<\|im_(start|end)\|>", re.I),
]

_INJECTION_ECHO = [
    re.compile(r"игнорир\w+\s+(все\s+)?(предыдущ\w+\s+)?(инструкции|правила|промпт)", re.I),
    re.compile(r"\bignore\s+(all\s+)?(previous|prior)\s+instructions", re.I),
    re.compile(r"\bнапиши\s+рецепт\b", re.I),
]

_PROFANITY = re.compile(
    r"\b("
    r"еба\w*|ёба\w*|ебан\w*|ебеш\w*|ебейш\w*|"
    r"бля\w*|"
    r"пизд\w*|"
    r"хуй|хуя|хуе\w*|хуё\w*|"
    r"мудак\w*|долбоёб\w*|долбоеб\w*"
    r")\b",
    re.I,
)

_THINK_BLOCK = re.compile(r"<think>.*?</think>", re.DOTALL)
_TRAILING_THINK = re.compile(r"<think>.*$", re.DOTALL)

_RECOMMEND_REFUSAL = (
    "Извините, я не даю медицинских рекомендаций — это компетенция врача. "
    "Я могу сообщить только информацию о наличии и цене услуг в каталоге."
)
_INJECTION_REFUSAL = (
    "Извините, я могу отвечать только на вопросы об услугах нашего медицинского центра."
)
_LANG_FALLBACK = (
    "Извините, в ответе произошёл сбой. Пожалуйста, переформулируйте вопрос — "
    "я отвечаю только на русском языке."
)
_TEMPLATE_FALLBACK = (
    "Извините, я не смог сформировать корректный ответ. "
    "Пожалуйста, переформулируйте вопрос."
)

_LATIN_DOMINANT = re.compile(r"[A-Za-z]")
_CYRILLIC = re.compile(r"[А-Яа-яЁё]")


def filter_output(text: str) -> str:
    if not text:
        return "Извините, я не смог сформировать ответ. Попробуйте переформулировать вопрос."

    text = _THINK_BLOCK.sub("", text)
    text = _TRAILING_THINK.sub("", text)
    text = text.strip()
    if not text:
        return "Извините, я не смог сформировать ответ. Попробуйте переформулировать вопрос."

    if _PROFANITY.search(text):
        return _TEMPLATE_FALLBACK

    for pat in _INJECTION_ECHO:
        if pat.search(text):
            return _INJECTION_REFUSAL

    for pat in _TEMPLATE_LEAKAGE:
        if pat.search(text):
            return _TEMPLATE_FALLBACK

    for pat in _RECOMMENDATION_PATTERNS:
        if pat.search(text):
            return _RECOMMEND_REFUSAL

    latin = len(_LATIN_DOMINANT.findall(text))
    cyrillic = len(_CYRILLIC.findall(text))
    if latin > 20 and latin > cyrillic * 2:
        return _LANG_FALLBACK

    text = text.replace("*", "")
    text = re.sub(r"  +", " ", text).strip()

    if len(text) > _MAX_LEN:
        text = text[: _MAX_LEN - 1] + "…"

    return text
