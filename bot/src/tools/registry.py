"""Tool registry: dispatches parsed tool calls to the right tool."""
from __future__ import annotations

import logging
from dataclasses import dataclass

from src.tools.search_memory import SearchMemoryTool
from src.tools.search_services import SearchServicesTool

log = logging.getLogger(__name__)


@dataclass
class ToolResult:
    name: str
    content: str
    ok: bool = True


class ToolRegistry:
    def __init__(
        self,
        *,
        search_services: SearchServicesTool,
        search_memory: SearchMemoryTool,
    ) -> None:
        self._search_services = search_services
        self._search_memory = search_memory

    @property
    def known_tools(self) -> set[str]:
        return {"search_services", "search_memory"}

    async def dispatch(self, *, chat_id: int, call: dict) -> ToolResult:
        name = call.get("tool")
        if name == "search_services":
            query = str(call.get("query") or "")
            top_k = int(call.get("top_k") or 5)
            out = await self._search_services.run(query=query, top_k=top_k)
            return ToolResult(name=name, content=out)
        if name == "search_memory":
            query = str(call.get("query") or "")
            top_k = int(call.get("top_k") or 3)
            out = await self._search_memory.run(chat_id=chat_id, query=query, top_k=top_k)
            return ToolResult(name=name, content=out)

        return ToolResult(
            name=name or "?",
            content=(
                f"Ошибка: неизвестный инструмент «{name}». "
                f"Доступные: search_services, search_memory. "
                "Если нужно ответить клиенту — пиши обычным текстом без JSON."
            ),
            ok=False,
        )
