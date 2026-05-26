"""Input safety filter.

Philosophy after testing: keep prompt-injection filters HARD (they have no
false-positive cost — nobody legitimately writes "ignore previous instructions"
to a service-catalog bot). Make medical-advice filters SOFT — only catch
unambiguous treatment-seeking phrasings. Broad keyword filters ("препарат от")
have false positives on catalog brand names ("препараты от ImmunoHealth") and
must be avoided. Let the output filter and the system prompt handle the rest.

Off-topic filters stay tight on a small handful of clearly-unrelated topics
(weather, jokes, currency exchange) so users get a quick "not my domain"
without burning an LLM round trip.
"""
from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass

_MAX_LEN = 500

# Hard prompt-injection blockers — strict, no soft-listing.
_INJECTION_PATTERNS = [
    re.compile(r"\bignore\s+(all\s+)?(previous|prior|above)\s+(instructions|rules|prompts)", re.I),
    re.compile(r"\b(забудь|игнорируй|игнорир)\w*\s+(все\s+)?(предыдущ\w+\s+)?(инструкции|правила|промпт)", re.I),
    re.compile(r"\bsystem\s*[:=]?\s*prompt", re.I),
    re.compile(r"\bсистемный\s+промпт", re.I),
    re.compile(r"act\s+as\s+(a\s+)?(dan|jailbreak|admin)", re.I),
    re.compile(r"jailbreak", re.I),
    re.compile(r"reveal\s+(your\s+)?(system|hidden)\s+(prompt|instructions)", re.I),
    re.compile(r"\bнапиши\s+(мне\s+)?рецепт\b", re.I),  # "напиши рецепт плова" — meta-injection
]

# Soft medical-advice blockers — only the clearest treatment-seeking phrasings.
# Symptom mentions ("у меня болит живот") are NOT blocked here: users
# legitimately describe complaints to ask which analyses to take. The bot's
# job (with the domain context in the system prompt) is to route them to
# relevant diagnostic services. Only outright "what should I take?" /
# "diagnose me" / "how do I treat X?" patterns get auto-refused.
_RECOMMENDATION_PATTERNS = [
    # Direct treatment-seeking — "что мне принять/выпить"
    re.compile(r"\bчто\s+(мне\s+)?(принять|принимать|пить|попить|выпить|пропить)\b", re.I),
    # Asking how to treat / cure
    re.compile(r"\bкак\s+(мне\s+)?(лечить|вылечить|избавиться)\b", re.I),
    # Asking for a diagnosis
    re.compile(r"\bпоставь\s+(мне\s+)?диагноз", re.I),
    re.compile(r"\bчто\s+(это|у\s+меня)\s+за\s+болезн", re.I),
    # "Is this dangerous?" — soliciting medical opinion
    re.compile(r"\bопасн[оаыи]\s+ли\s+это", re.I),
    # Direct "what would you choose / recommend?" — asks bot for opinion
    re.compile(r"\bкак(ой|ую|ое|ие)\s+(бы\s+)?ты\s+(\w+\s+){0,3}(выбрал|посовет|рекоменд|предлож)", re.I),
    re.compile(r"\bты\s+(\w+\s+){0,2}(посовет|порекоменд|рекоменду)\w*", re.I),
    re.compile(r"\bстоит\s+ли\s+мне\b", re.I),
    re.compile(r"\bнужно\s+ли\s+мне\b", re.I),
    # Imperative "advise me"
    re.compile(r"^\s*посовет\w*(\s|$)", re.I),
    re.compile(r"^\s*порекоменд\w*(\s|$)", re.I),
]

# Off-topic — clearly unrelated to the clinic.
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
