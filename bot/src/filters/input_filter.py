"""Фильтр пользовательского ввода: блокирует prompt-инъекции, прямые запросы лечения и явные оффтоп-темы до вызова LLM."""
from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass

_MAX_LEN = 500

_INJECTION_PATTERNS = [
    re.compile(r"\bignore\s+(all\s+)?(previous|prior|above)\s+(instructions|rules|prompts)", re.I),
    re.compile(r"\b(забудь|игнорируй|игнорир)\w*\s+(все\s+)?(предыдущ\w+\s+)?(инструкции|правила|промпт)", re.I),
    re.compile(r"\bsystem\s*[:=]?\s*prompt", re.I),
    re.compile(r"\bсистемный\s+промпт", re.I),
    re.compile(r"act\s+as\s+(a\s+)?(dan|jailbreak|admin)", re.I),
    re.compile(r"jailbreak", re.I),
    re.compile(r"reveal\s+(your\s+)?(system|hidden)\s+(prompt|instructions)", re.I),
    re.compile(r"\bнапиши\s+(мне\s+)?рецепт\b", re.I),
]

_RECOMMENDATION_PATTERNS = [
    re.compile(r"\bчто\s+(мне\s+)?(принять|принимать|пить|попить|выпить|пропить)\b", re.I),
    re.compile(r"\bкак\s+(мне\s+)?(лечить|вылечить|избавиться)\b", re.I),
    re.compile(r"\bпоставь\s+(мне\s+)?диагноз", re.I),
    re.compile(r"\bчто\s+(это|у\s+меня)\s+за\s+болезн", re.I),
    re.compile(r"\bопасн[оаыи]\s+ли\s+это", re.I),
    re.compile(r"\bкак(ой|ую|ое|ие)\s+(бы\s+)?ты\s+(\w+\s+){0,3}(выбрал|посовет|рекоменд|предлож)", re.I),
    re.compile(r"\bты\s+(\w+\s+){0,2}(посовет|порекоменд|рекоменду)\w*", re.I),
    re.compile(r"\bстоит\s+ли\s+мне\b", re.I),
    re.compile(r"\bнужно\s+ли\s+мне\b", re.I),
    re.compile(r"^\s*посовет\w*(\s|$)", re.I),
    re.compile(r"^\s*порекоменд\w*(\s|$)", re.I),
]

_OFFTOPIC_PATTERNS = [
    re.compile(r"\bпогод[аы]\b", re.I),
    re.compile(r"\bкурс\s+(доллар|евро|валют)", re.I),
    re.compile(r"\bновост[ия]\b", re.I),
    re.compile(r"\bрасскажи\s+анекдот", re.I),
]

_REFUSAL_INJECTION = (
    "Извините, я могу отвечать только на вопросы об услугах нашего медицинского центра."
)
_REFUSAL_RECOMMEND = (
    "Я не даю медицинских рекомендаций — обратитесь, пожалуйста, к врачу. "
    "Я могу подсказать, есть ли услуга в нашем каталоге и сколько она стоит."
)
_REFUSAL_OFFTOPIC = (
    "Я отвечаю только на вопросы об услугах нашего медицинского центра."
)


@dataclass
class FilterResult:
    passed: bool
    cleaned: str
    refusal: str | None = None


def filter_input(text: str) -> FilterResult:
    if text is None:
        return FilterResult(False, "", _REFUSAL_INJECTION)

    text = "".join(
        ch for ch in text if ch in ("\n", "\t") or unicodedata.category(ch)[0] != "C"
    )
    text = re.sub(r"\s+", " ", text).strip()

    if not text:
        return FilterResult(False, "", "Пустой запрос. Задайте, пожалуйста, вопрос об услуге.")

    if len(text) > _MAX_LEN:
        text = text[:_MAX_LEN]

    for pat in _INJECTION_PATTERNS:
        if pat.search(text):
            return FilterResult(False, text, _REFUSAL_INJECTION)

    for pat in _RECOMMENDATION_PATTERNS:
        if pat.search(text):
            return FilterResult(False, text, _REFUSAL_RECOMMEND)

    for pat in _OFFTOPIC_PATTERNS:
        if pat.search(text):
            return FilterResult(False, text, _REFUSAL_OFFTOPIC)

    return FilterResult(True, text, None)
