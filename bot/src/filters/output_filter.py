from __future__ import annotations

import re

_MAX_LEN = 4000  # Telegram limit is 4096

# Recommendation/advice phrasing — soft list. Only catch first-/second-person
# direct recommendations. Neutral uses ("врач рекомендует", "рекомендуется
# обратиться к администратору") are allowed through — they're legitimate.
_RECOMMENDATION_PATTERNS = [
    # First-person: "я рекомендую", "я посоветую"
    re.compile(r"\bя\s+(вам\s+|тебе\s+)?рекоменду[ею]\b", re.I),
    re.compile(r"\bя\s+(вам\s+|тебе\s+)?(посоветую|порекомендую)\b", re.I),
    # Direct second-person: "рекомендую вам/тебе X", "вам рекомендую"
    re.compile(r"\bрекоменду[ею]\s+(вам|тебе)\b", re.I),
    re.compile(r"\b(вам|тебе)\s+рекоменду[ею]\b", re.I),
    # Modal advice: "вам стоит/следует/нужно/лучше"
    re.compile(r"\bвам\s+(стоит|следует|нужно|лучше)\b", re.I),
    # Suggestion templates
    re.compile(r"\bя\s+бы\s+(вам\s+)?(посоветовал|выбрал|порекомендовал|предложил)", re.I),
    re.compile(r"\bлучше\s+(всего\s+)?(выбрать|подойдёт|подходит)\b", re.I),
    re.compile(r"\bна\s+ваш(ем)?\s+месте\b", re.I),
    re.compile(r"\bстоит\s+(попробовать|выбрать|взять|пройти|сделать)\b", re.I),
]

# Template/structural leakage — the model echoed the prompt scaffold.
_TEMPLATE_LEAKAGE = [
    re.compile(r"(?:^|[\n.])\s*каталог\s*:", re.I),
    re.compile(r"(?:^|[\n.])\s*клиент\s*:", re.I),
    re.compile(r"(?:^|[\n.])\s*вопрос\s*:", re.I),
    re.compile(r"(?:^|[\n.])\s*ответ\s*:", re.I),
    re.compile(r"(?:^|[\n.])\s*system\s*:", re.I),
    re.compile(r"<\|im_(start|end)\|>", re.I),
]

# Injection echo — model regurgitated adversarial-style strings.
_INJECTION_ECHO = [
    re.compile(r"игнорир\w+\s+(все\s+)?(предыдущ\w+\s+)?(инструкции|правила|промпт)", re.I),
    re.compile(r"\bignore\s+(all\s+)?(previous|prior)\s+instructions", re.I),
    re.compile(r"\bнапиши\s+рецепт\b", re.I),
]

# Crude profanity guard — covers the most common Russian stems.
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
_TRAILING_THINK = re.compile(r"<think>.*$", re.DOTALL)  # truncated thinking block

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

    # Strip all '*' characters — the LLM likes to emit **bold** / *italic*
    # markdown, but we send to Telegram as plain text, so asterisks render
    # literally and look broken. Cheaper than parsing markdown.
    text = text.replace("*", "")
    # Collapse the double spaces that "**foo**" → "foo" can leave behind
    text = re.sub(r"  +", " ", text).strip()

    if len(text) > _MAX_LEN:
        text = text[: _MAX_LEN - 1] + "…"

    return text
