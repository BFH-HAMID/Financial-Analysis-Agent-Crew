"""Shared helpers for building and running agno agents."""

from __future__ import annotations

from typing import Any

from utils.logger import get_logger

logger = get_logger(__name__)


def response_text(response: Any) -> str:
    """Extract plain Markdown text from an agno ``RunOutput``/``TeamRunOutput``.

    Handles the shapes agno has used across versions: an object with
    ``get_content_as_string()``, an object with a ``content`` attribute, or a
    bare string.
    """
    if response is None:
        return ""
    if isinstance(response, str):
        return response.strip()

    getter = getattr(response, "get_content_as_string", None)
    if callable(getter):
        try:
            text = getter()
            if text:
                return str(text).strip()
        except Exception as exc:  # noqa: BLE001
            logger.debug("get_content_as_string failed: %s", exc)

    content = getattr(response, "content", response)
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict):
                parts.append(str(item.get("text") or item.get("content") or ""))
            else:
                parts.append(str(getattr(item, "text", "") or ""))
        return "\n".join(part for part in parts if part).strip()
    return str(content or "").strip()


def agent_summary(agent: Any) -> dict[str, str]:
    """Return display metadata for an agno agent."""
    return {
        "name": str(getattr(agent, "name", "agent")),
        "role": str(getattr(agent, "role", "") or ""),
    }


def tool_names(agent: Any) -> list[str]:
    """Best-effort list of tool names attached to an agent."""
    names: list[str] = []
    for tool in getattr(agent, "tools", []) or []:
        name = getattr(tool, "name", None) or getattr(tool, "__name__", None)
        if name:
            names.append(str(name))
    return names


__all__ = ["agent_summary", "response_text", "tool_names"]
