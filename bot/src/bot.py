from __future__ import annotations

import logging

from aiogram import Bot, Dispatcher, F
from aiogram.filters import CommandStart
from aiogram.types import Message

from src.llm.prompts import GREETING
from src.pipeline import Pipeline

log = logging.getLogger(__name__)


def build_dispatcher(pipeline: Pipeline) -> Dispatcher:
    dp = Dispatcher()

    @dp.message(CommandStart())
    async def on_start(message: Message) -> None:
        await message.answer(GREETING)
        # Record the greeting so the next text message is not treated as a new chat
        await pipeline.mark_greeted(message.chat.id)

    @dp.message(F.text)
    async def on_text(message: Message) -> None:
        chat_id = message.chat.id
        text = message.text or ""
        log.info("incoming chat=%s len=%d", chat_id, len(text))
        is_new = await pipeline.is_new_chat(chat_id)
        if is_new:
            await message.answer(GREETING)
            await pipeline.mark_greeted(chat_id)
        try:
            reply = await pipeline.handle(chat_id, text)
        except Exception:
            log.exception("pipeline failed")
            reply = (
                "Извините, произошла внутренняя ошибка. "
                "Попробуйте, пожалуйста, ещё раз позже."
            )
        await message.answer(reply)

    @dp.message()
    async def on_non_text(message: Message) -> None:
        await message.answer(
            "Я обрабатываю только текстовые сообщения. "
            "Пожалуйста, опишите ваш вопрос словами."
        )

    return dp


def build_bot(token: str) -> Bot:
    return Bot(token=token)
