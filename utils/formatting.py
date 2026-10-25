"""Formatting helpers shared by the Streamlit UI and the PDF generator."""

from __future__ import annotations

from typing import Any

NOT_AVAILABLE = "n/a"


def safe_float(value: Any) -> float | None:
    """Return ``value`` as a finite float, or ``None``."""
    if value is None:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if number != number or number in (float("inf"), float("-inf")):
        return None
    return number


def fmt_number(value: Any, decimals: int = 2) -> str:
    """Format a ratio/score such as ``P/E = 28.412`` → ``"28.41"``."""
    number = safe_float(value)
    if number is None:
        return NOT_AVAILABLE
    return f"{number:,.{decimals}f}"


def fmt_pct(value: Any, decimals: int = 2, signed: bool = True) -> str:
    """Format a percentage value already expressed in percent units."""
    number = safe_float(value)
    if number is None:
        return NOT_AVAILABLE
    sign = "+" if signed and number > 0 else ""
    return f"{sign}{number:,.{decimals}f}%"


def fmt_currency(value: Any, decimals: int = 2, symbol: str = "$") -> str:
    """Format a price-like value as currency."""
    number = safe_float(value)
    if number is None:
        return NOT_AVAILABLE
    return f"{symbol}{number:,.{decimals}f}"


def fmt_compact(value: Any, symbol: str = "$") -> str:
    """Format large numbers in compact notation (market cap, volume)."""
    number = safe_float(value)
    if number is None:
        return NOT_AVAILABLE
    magnitude = abs(number)
    sign = "-" if number < 0 else ""
    if magnitude >= 1e12:
        return f"{sign}{symbol}{magnitude / 1e12:,.2f}T"
    if magnitude >= 1e9:
        return f"{sign}{symbol}{magnitude / 1e9:,.2f}B"
    if magnitude >= 1e6:
        return f"{sign}{symbol}{magnitude / 1e6:,.2f}M"
    if magnitude >= 1e3:
        return f"{sign}{symbol}{magnitude / 1e3:,.2f}K"
    return f"{sign}{symbol}{magnitude:,.2f}"


def fmt_ratio(value: Any) -> str:
    """Format a unitless ratio with two decimals."""
    return fmt_number(value, 2)


def fmt_confidence(value: Any) -> str:
    """Format a 0-100 confidence score."""
    number = safe_float(value)
    if number is None:
        return NOT_AVAILABLE
    return f"{number:.0f}/100"


def sentiment_emoji(label: str | None) -> str:
    """Map a sentiment label to an emoji for the UI."""
    mapping = {
        "bullish": "🟢",
        "positive": "🟢",
        "neutral": "🟡",
        "mixed": "🟡",
        "negative": "🔴",
        "bearish": "🔴",
    }
    return mapping.get((label or "").lower(), "⚪")


__all__ = [
    "NOT_AVAILABLE",
    "fmt_compact",
    "fmt_confidence",
    "fmt_currency",
    "fmt_number",
    "fmt_pct",
    "fmt_ratio",
    "safe_float",
    "sentiment_emoji",
]
