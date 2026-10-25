"""Logging helpers and the Agent Conversation Log.

Two concerns live here:

1. Standard application logging (rotating file + console) via :func:`get_logger`.
2. A structured, thread-safe :class:`ConversationLog` that records every agent
   hand-off, tool call and revision round so the UI can show an
   "Agent Conversation Log" expander and the PDF can include an audit trail.
"""

from __future__ import annotations

import json
import logging
import threading
import time
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Any

from config.settings import get_settings

_LOG_FORMAT = "%(asctime)s | %(levelname)-8s | %(name)-28s | %(message)s"
_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"
_CONFIGURED = False
_CONFIG_LOCK = threading.Lock()


def _configure_root_logging() -> None:
    """Attach console + rotating file handlers once per process."""
    global _CONFIGURED
    with _CONFIG_LOCK:
        if _CONFIGURED:
            return

        settings = get_settings()
        settings.ensure_directories()

        root = logging.getLogger("financial_crew")
        root.setLevel(getattr(logging, settings.log_level, logging.INFO))
        root.propagate = False

        if not root.handlers:
            formatter = logging.Formatter(_LOG_FORMAT, datefmt=_DATE_FORMAT)

            console = logging.StreamHandler()
            console.setFormatter(formatter)
            root.addHandler(console)

            try:
                file_handler = RotatingFileHandler(
                    settings.log_file,
                    maxBytes=settings.max_log_bytes,
                    backupCount=settings.log_backup_count,
                    encoding="utf-8",
                )
                file_handler.setFormatter(formatter)
                root.addHandler(file_handler)
            except OSError:  # read-only filesystem, e.g. some serverless hosts
                root.warning("File logging disabled (cannot write %s)", settings.log_file)

        _CONFIGURED = True


def get_logger(name: str = "financial_crew") -> logging.Logger:
    """Return a namespaced logger that writes to console and a rotating file.

    Args:
        name: Usually ``__name__``. Names are grouped under ``financial_crew``.
    """
    _configure_root_logging()
    if name == "financial_crew" or name.startswith("financial_crew."):
        return logging.getLogger(name)
    return logging.getLogger(f"financial_crew.{name}")


@dataclass
class LogEvent:
    """One entry in the agent conversation log."""

    timestamp: str
    agent: str
    event: str
    message: str
    duration_ms: float | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def to_markdown(self) -> str:
        """Render the event as a single Markdown bullet."""
        stamp = f"`{self.timestamp}`"
        who = f"**{self.agent}**"
        kind = f"_{self.event}_"
        duration = f" ({self.duration_ms:.0f} ms)" if self.duration_ms else ""
        body = self.message.strip().replace("\n", "\n  ")
        return f"- {stamp} {who} {kind}{duration}: {body}"


class ConversationLog:
    """In-memory, thread-safe record of the multi-agent conversation.

    The log is deliberately transport-agnostic: the Streamlit UI renders it with
    :meth:`to_markdown`, the PDF generator uses :meth:`to_records`, and JSON
    exports use :meth:`to_dict`.
    """

    def __init__(self, max_events: int = 500) -> None:
        self._events: list[LogEvent] = []
        self._lock = threading.Lock()
        self._max_events = max_events
        self._started = time.perf_counter()

    # ------------------------------------------------------------- recording
    def add(
        self,
        agent: str,
        event: str,
        message: str,
        duration_ms: float | None = None,
        **metadata: Any,
    ) -> LogEvent:
        """Append an event and mirror it into the application logger."""
        entry = LogEvent(
            timestamp=datetime.now(timezone.utc).strftime("%H:%M:%S"),
            agent=agent,
            event=event,
            message=message,
            duration_ms=round(duration_ms, 1) if duration_ms is not None else None,
            metadata=metadata,
        )
        with self._lock:
            self._events.append(entry)
            if len(self._events) > self._max_events:
                self._events = self._events[-self._max_events :]
        get_logger("financial_crew.conversation").info("[%s] %s: %s", agent, event, message)
        return entry

    def agent_start(self, agent: str, message: str) -> LogEvent:
        return self.add(agent, "started", message)

    def agent_done(self, agent: str, message: str, duration_s: float) -> LogEvent:
        return self.add(agent, "finished", message, duration_ms=duration_s * 1000)

    def tool_call(self, agent: str, tool: str, message: str, **metadata: Any) -> LogEvent:
        return self.add(agent, "tool", f"{tool} — {message}", **metadata)

    def warning(self, agent: str, message: str) -> LogEvent:
        return self.add(agent, "warning", message)

    def error(self, agent: str, message: str) -> LogEvent:
        return self.add(agent, "error", message)

    @contextmanager
    def timed(
        self, agent: str, event: str = "started", message: str = ""
    ) -> Iterator[dict[str, Any]]:
        """Context manager that logs start/finish and measures wall time.

        Example
        -------
        >>> with log.timed("Data Analyst", message="Computing KPIs") as t:
        ...     compute()
        >>> t["duration_s"] > 0
        True
        """
        started = time.perf_counter()
        if message:
            self.add(agent, event, message)
        info: dict[str, Any] = {}
        try:
            yield info
        finally:
            elapsed = time.perf_counter() - started
            info["duration_s"] = elapsed
            self.add(agent, "finished", f"completed in {elapsed:.1f}s", duration_ms=elapsed * 1000)

    # --------------------------------------------------------------- reading
    @property
    def events(self) -> list[LogEvent]:
        with self._lock:
            return list(self._events)

    def __len__(self) -> int:
        with self._lock:
            return len(self._events)

    def to_records(self) -> list[dict[str, Any]]:
        return [event.to_dict() for event in self.events]

    def to_dict(self) -> dict[str, Any]:
        return {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "event_count": len(self),
            "total_duration_s": round(time.perf_counter() - self._started, 2),
            "events": self.to_records(),
        }

    def to_markdown(self) -> str:
        """Render the whole log as Markdown for the Streamlit expander."""
        events = self.events
        if not events:
            return "_No agent activity recorded yet._"
        lines = ["### 🗣️ Agent Conversation Log", ""]
        lines.extend(event.to_markdown() for event in events)
        return "\n".join(lines)

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, ensure_ascii=False)

    def save(self, path: str | Path) -> Path:
        """Persist the conversation log as JSON next to the generated report."""
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(self.to_json(), encoding="utf-8")
        return destination


__all__ = ["ConversationLog", "LogEvent", "get_logger"]
