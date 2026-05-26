from __future__ import annotations

import re

_MAX_LEN = 4000  # Telegram limit is 4096

# Recommendation/advice phrasing — narrowed to MEDICATION/TREATMENT advice
# only. Diagnostic suggestions ("вам стоит сдать ферритин и ТТГ", "рекомендую
# пройти УЗИ", "я бы посоветовала чек-ап Кардио") are legitimate consultative
# behaviour and pass through. Only patterns explicitly tied to drug/treatment
# verbs (пить / принимать / выпить / купить лекарство / лечиться) are blocked.
_MED_VERBS = (
    r"(?:"
    r"пи[лть]\w*"            # пить / пил / пила / пили / пить-be
    r"|пь[еёюя]\w*"          # пьёт / пьют / пью
    r"|пей(?:те)?"           # пей / пейте
    r"|вы?пи[лть]\w*"        # выпил / выпила / выпили / выпить
    r"|вы?пь[еёюя]\w*"       # выпьет / выпью
    r"|вы?пей(?:те)?"        # выпей / выпейте
    r"|попи[лть]\w*"
    r"|пропи[лть]\w*"
    r"|приня[лт]\w*"         # принял / принять / приняла
    r"|принима[йлтю]\w*"     # принимать / принимала / принимаю / принимайте
    r"|принима\w*"           # safety net
    r"|купи\w+\s+лекарств\w+"
    r"|лечи(?:ть|ться|тесь)\w*"
    r")"
)

_RECOMMENDATION_PATTERNS = [
    # "Рекомендую (вам/тебе) пить/принимать..." → drug recommendation
    re.compile(r"\bрекоменду[ею]\s+(вам\s+|тебе\s+)?" + _MED_VERBS, re.I),
    # "Я рекомендую/посоветую/порекомендую (вам/тебе) пить..."
    re.compile(r"\bя\s+(вам\s+|тебе\s+)?(рекоменду[ею]|посоветую|порекомендую)\s+.{0,20}" + _MED_VERBS, re.I),
    # "Вам стоит/лучше/следует/нужно пить/принимать..."
    re.compile(r"\b(вам|тебе)\s+(стоит|лучше|следует|нужно)\s+" + _MED_VERBS, re.I),
    # "Я бы (вам) посоветовал(а) пить/принимать..."
    re.compile(r"\bя\s+бы\s+(вам\s+|тебе\s+)?(посоветовал[аи]?|порекомендовал[аи]?)\s+.{0,20}" + _MED_VERBS, re.I),
    # "На вашем месте я бы выпил/принял..."
    re.compile(r"\bна\s+ваш(ем)?\s+месте\s+.{0,30}" + _MED_VERBS, re.I),
    # Direct treatment-recommending: "принимайте/пейте N таблеток"
    re.compile(r"\b(принимайте|пейте|выпейте)\s+\w+\s+таблет\w*", re.I),
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
