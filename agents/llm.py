"""Gemini model factory shared by all agents.

The crew uses a single :class:`agno.models.google.Gemini` instance so that model
name, temperature and retry policy are configured in exactly one place.
"""

from __future__ import annotations

from typing import Any

from config.settings import get_settings
from utils.logger import get_logger

logger = get_logger(__name__)


class LLMConfigurationError(RuntimeError):
    """Raised when the language model cannot be configured."""


def build_model(*, temperature: float | None = None, **overrides: Any) -> Any:
    """Create a configured Gemini model instance.

    Args:
        temperature: Optional sampling temperature override.
        **overrides: Extra keyword arguments forwarded to ``Gemini``.

    Returns:
        A ready-to-use ``agno.models.google.Gemini`` instance.

    Raises:
        LLMConfigurationError: If ``GOOGLE_API_KEY`` is missing or the agno
            Google provider is unavailable.
    """
    settings = get_settings()
    if not settings.google_api_key:
        raise LLMConfigurationError(
            "GOOGLE_API_KEY is not set. Add it to your .env file or environment "
            "(get a key at https://aistudio.google.com/app/apikey)."
        )

    try:
        from agno.models.google import Gemini
    except ImportError as exc:  # pragma: no cover - agno is a hard dependency
        raise LLMConfigurationError(
            "The agno Google provider is not installed. Run `pip install -r requirements.txt`."
        ) from exc

    kwargs: dict[str, Any] = {
        "id": settings.gemini_model,
        "api_key": settings.google_api_key,
        "temperature": settings.gemini_temperature if temperature is None else temperature,
        "timeout": settings.llm_timeout_seconds,
        "retries": 1,
    }
    kwargs.update(overrides)

    try:
        model = Gemini(**kwargs)
    except Exception as exc:  # noqa: BLE001
        raise LLMConfigurationError(f"Could not initialise Gemini: {exc}") from exc

    logger.info(
        "Gemini model ready: %s (temperature=%s)", settings.gemini_model, kwargs["temperature"]
    )
    return model


def llm_is_configured() -> bool:
    """Return True when an API key is present (does not validate the key)."""
    return get_settings().llm_available


__all__ = ["LLMConfigurationError", "build_model", "llm_is_configured"]
