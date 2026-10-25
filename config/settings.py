"""Central configuration for the Financial Analysis Agent Crew.

All runtime configuration is loaded from environment variables (optionally via a
``.env`` file at the project root).  Import :func:`get_settings` instead of
reading ``os.environ`` directly so that every module sees the same values.

Example
-------
>>> from config.settings import get_settings
>>> settings = get_settings()
>>> settings.gemini_model
'gemini-2.5-flash'
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any

try:  # python-dotenv is optional at import time so tests can inject env vars
    from dotenv import load_dotenv
except ImportError:  # pragma: no cover - dependency is in requirements.txt
    load_dotenv = None  # type: ignore[assignment]


BASE_DIR = Path(__file__).resolve().parent.parent
ENV_FILE = BASE_DIR / ".env"

if load_dotenv is not None:
    # ``override=False`` keeps real environment variables (Docker, CI, Streamlit
    # Cloud secrets) authoritative over the local .env file.
    load_dotenv(dotenv_path=ENV_FILE, override=False)


def _get_bool(name: str, default: bool = False) -> bool:
    """Read a boolean-ish environment variable."""
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "y", "on"}


def _get_int(name: str, default: int) -> int:
    """Read an integer environment variable, falling back on bad input."""
    raw = os.getenv(name)
    if raw is None or not raw.strip():
        return default
    try:
        return int(raw)
    except ValueError:
        return default


def _get_float(name: str, default: float) -> float:
    """Read a float environment variable, falling back on bad input."""
    raw = os.getenv(name)
    if raw is None or not raw.strip():
        return default
    try:
        return float(raw)
    except ValueError:
        return default


@dataclass(frozen=True)
class Settings:
    """Immutable application settings."""

    # ------------------------------------------------------------------ paths
    base_dir: Path = BASE_DIR
    reports_dir: Path = BASE_DIR / "reports" / "output"
    chart_dir: Path = BASE_DIR / "reports" / "output" / "charts"
    pdf_dir: Path = BASE_DIR / "reports" / "output" / "pdfs"
    log_dir: Path = BASE_DIR / "logs"

    # ---------------------------------------------------------------- secrets
    google_api_key: str | None = None
    news_api_key: str | None = None

    # ---------------------------------------------------------------- Gemini
    gemini_model: str = "gemini-2.5-flash"
    gemini_temperature: float = 0.3
    llm_timeout_seconds: int = 120

    # ----------------------------------------------------------------- agents
    max_revision_rounds: int = 2
    team_mode: str = "coordinate"
    team_max_iterations: int = 6
    show_members_responses: bool = True
    sentiment_backend: str = "lexicon"  # "lexicon" | "llm"

    # ------------------------------------------------------------------- data
    risk_free_rate: float = 0.04
    trading_days: int = 252
    max_news_results: int = 8
    news_lookback_days: int = 30
    request_timeout_seconds: int = 20

    # --------------------------------------------------------------- fallback
    # Demo mode generates *clearly labelled synthetic* data when live market /
    # news providers are unreachable (offline demo, restricted network, CI).
    allow_demo_data: bool = True
    force_demo_data: bool = False
    allow_offline_llm_fallback: bool = True

    # ----------------------------------------------------------------- charts
    chart_width: int = 1100
    chart_height: int = 550
    chart_scale: float = 2.0

    # ---------------------------------------------------------------- logging
    log_level: str = "INFO"
    log_file: Path = BASE_DIR / "logs" / "financial_crew.log"
    max_log_bytes: int = 2_000_000
    log_backup_count: int = 3

    # --------------------------------------------------------------- metadata
    app_name: str = "Financial Analysis Agent Crew"
    app_version: str = "1.0.0"
    extra: dict[str, Any] = field(default_factory=dict)

    # ------------------------------------------------------------ properties
    @property
    def llm_available(self) -> bool:
        """True when a Gemini API key is configured."""
        return bool(self.google_api_key)

    @property
    def demo_mode(self) -> bool:
        """True when the user explicitly forced demo mode."""
        return self.force_demo_data

    def ensure_directories(self) -> None:
        """Create every directory the application writes to."""
        for path in (
            self.reports_dir,
            self.chart_dir,
            self.pdf_dir,
            self.log_dir,
        ):
            path.mkdir(parents=True, exist_ok=True)

    def public_dict(self) -> dict[str, Any]:
        """Return a safe-to-display summary (never exposes secrets)."""
        return {
            "app_name": self.app_name,
            "app_version": self.app_version,
            "model": self.gemini_model,
            "llm_available": self.llm_available,
            "sentiment_backend": self.sentiment_backend,
            "max_revision_rounds": self.max_revision_rounds,
            "allow_demo_data": self.allow_demo_data,
            "force_demo_data": self.force_demo_data,
            "news_enabled": True,
            "news_api_key_set": bool(self.news_api_key),
        }

    # ------------------------------------------------------------- factories
    @classmethod
    def from_env(cls) -> Settings:
        """Build settings from the current process environment."""
        base = BASE_DIR
        reports_dir = Path(os.getenv("REPORTS_DIR", str(base / "reports" / "output")))

        return cls(
            base_dir=base,
            reports_dir=reports_dir,
            chart_dir=reports_dir / "charts",
            pdf_dir=reports_dir / "pdfs",
            log_dir=Path(os.getenv("LOG_DIR", str(base / "logs"))),
            google_api_key=os.getenv("GOOGLE_API_KEY") or None,
            news_api_key=os.getenv("NEWS_API_KEY") or None,
            gemini_model=os.getenv("GEMINI_MODEL", "gemini-2.5-flash"),
            gemini_temperature=_get_float("GEMINI_TEMPERATURE", 0.3),
            llm_timeout_seconds=_get_int("LLM_TIMEOUT_SECONDS", 120),
            max_revision_rounds=_get_int("MAX_REVISION_ROUNDS", 2),
            team_mode=os.getenv("TEAM_MODE", "coordinate"),
            team_max_iterations=_get_int("TEAM_MAX_ITERATIONS", 6),
            show_members_responses=_get_bool("SHOW_MEMBERS_RESPONSES", True),
            sentiment_backend=os.getenv("SENTIMENT_BACKEND", "lexicon").lower(),
            risk_free_rate=_get_float("RISK_FREE_RATE", 0.04),
            trading_days=_get_int("TRADING_DAYS", 252),
            max_news_results=_get_int("MAX_NEWS_RESULTS", 8),
            news_lookback_days=_get_int("NEWS_LOOKBACK_DAYS", 30),
            request_timeout_seconds=_get_int("REQUEST_TIMEOUT_SECONDS", 20),
            allow_demo_data=_get_bool("ALLOW_DEMO_DATA", True),
            force_demo_data=_get_bool("FORCE_DEMO_DATA", False),
            allow_offline_llm_fallback=_get_bool("ALLOW_OFFLINE_LLM_FALLBACK", True),
            chart_width=_get_int("CHART_WIDTH", 1100),
            chart_height=_get_int("CHART_HEIGHT", 550),
            chart_scale=_get_float("CHART_SCALE", 2.0),
            log_level=os.getenv("LOG_LEVEL", "INFO").upper(),
            log_file=Path(os.getenv("LOG_FILE", str(base / "logs" / "financial_crew.log"))),
            max_log_bytes=_get_int("MAX_LOG_BYTES", 2_000_000),
            log_backup_count=_get_int("LOG_BACKUP_COUNT", 3),
            app_name=os.getenv("APP_NAME", "Financial Analysis Agent Crew"),
            app_version=os.getenv("APP_VERSION", "1.0.0"),
        )


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return the cached application settings."""
    settings = Settings.from_env()
    settings.ensure_directories()
    return settings


def reload_settings() -> Settings:
    """Clear the cache and re-read the environment (useful in tests)."""
    get_settings.cache_clear()
    return get_settings()


#: Convenient module-level handle. Import ``get_settings`` if you need a fresh one.
settings = get_settings()


__all__ = [
    "BASE_DIR",
    "ENV_FILE",
    "Settings",
    "get_settings",
    "reload_settings",
    "settings",
]
