"""Text parsing helpers for LLM output and user input validation."""

from __future__ import annotations

import json
import re
from datetime import date, datetime
from typing import Any

#: Symbols accepted by Yahoo Finance: equities (AAPL, BRK-B, 0700.HK),
#: indices (^GSPC, ^IXIC) and futures (ES=F).
TICKER_PATTERN = re.compile(r"^\^?[A-Z0-9][A-Z0-9.\-=]{0,14}$")
VALID_RECOMMENDATIONS = ("STRONG BUY", "BUY", "HOLD", "SELL", "STRONG SELL")


class InputValidationError(ValueError):
    """Raised when the user supplies an unusable ticker or date range."""


def normalize_ticker(ticker: str) -> str:
    """Clean and validate a stock ticker.

    Args:
        ticker: Raw user input, e.g. ``" aapl "``, ``"brk-b"`` or ``"^gspc"``.

    Returns:
        Upper-case ticker such as ``"AAPL"``, ``"BRK-B"`` or ``"^GSPC"``.

    Raises:
        InputValidationError: If the ticker is empty or contains illegal characters.
    """
    cleaned = (ticker or "").strip().upper().replace(" ", "")
    if not cleaned:
        raise InputValidationError("Please enter a stock ticker, e.g. AAPL or TSLA.")
    if not TICKER_PATTERN.match(cleaned):
        raise InputValidationError(
            f"'{ticker}' does not look like a valid ticker. Use letters, numbers, "
            "'.', '-' or a leading '^' (e.g. AAPL, BRK-B, 0700.HK, ^GSPC)."
        )
    return cleaned


def validate_date_range(start: date | str, end: date | str) -> tuple[date, date]:
    """Validate and normalise an analysis date range.

    Returns:
        A ``(start, end)`` tuple of ``datetime.date`` objects.

    Raises:
        InputValidationError: If dates are unparseable, reversed, in the future,
            or the window is shorter than five days.
    """
    start_date = _coerce_date(start, "start")
    end_date = _coerce_date(end, "end")

    if start_date >= end_date:
        raise InputValidationError("The start date must be before the end date.")
    if end_date > date.today():
        end_date = date.today()
    if (end_date - start_date).days < 5:
        raise InputValidationError("Please choose a date range of at least 5 days.")
    return start_date, end_date


def _coerce_date(value: date | str, label: str) -> date:
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    try:
        return datetime.strptime(str(value), "%Y-%m-%d").date()
    except (TypeError, ValueError) as exc:
        raise InputValidationError(f"Invalid {label} date: {value!r}") from exc


def extract_json_object(text: str) -> dict[str, Any] | None:
    """Extract the first JSON object from an LLM response.

    Handles fenced code blocks (```` ```json ... ``` ````) and prose surrounding
    the JSON. Returns ``None`` when no parseable object is found.
    """
    if not text:
        return None

    fenced = re.findall(r"```(?:json)?\s*(\{.*?\})\s*```", text, flags=re.DOTALL)
    candidates = list(fenced)

    # Fall back to balancing braces so nested objects survive.
    depth = 0
    start_index = -1
    for index, char in enumerate(text):
        if char == "{":
            if depth == 0:
                start_index = index
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0 and start_index >= 0:
                candidates.append(text[start_index : index + 1])
                start_index = -1

    for candidate in candidates:
        try:
            parsed = json.loads(candidate)
        except (json.JSONDecodeError, TypeError):
            continue
        if isinstance(parsed, dict):
            return parsed
    return None


def parse_recommendation(text: str) -> dict[str, Any]:
    """Extract a recommendation, confidence score and rationale from a report.

    Returns a dict with keys ``recommendation``, ``confidence``, ``rationale`` and
    ``found``. Missing values are returned as ``None`` rather than raising.
    """
    result: dict[str, Any] = {
        "recommendation": None,
        "confidence": None,
        "rationale": None,
        "found": False,
    }
    if not text:
        return result

    payload = extract_json_object(text) or {}
    for key in ("recommendation", "action", "rating"):
        value = payload.get(key) or payload.get(key.title())
        if isinstance(value, str):
            result["recommendation"] = _normalize_recommendation(value)
            break

    for key in ("confidence", "confidence_score", "confidence_pct"):
        value = payload.get(key)
        if value is not None:
            result["confidence"] = _clamp_confidence(value)
            break

    rationale = payload.get("rationale") or payload.get("confidence_rationale")
    if isinstance(rationale, str):
        result["rationale"] = rationale.strip()

    if result["recommendation"] is None:
        result["recommendation"] = _parse_recommendation_from_prose(text)
    if result["confidence"] is None:
        result["confidence"] = _parse_confidence_from_prose(text)
    if result["rationale"] is None:
        result["rationale"] = extract_markdown_section(
            text, ["confidence rationale", "why this confidence", "confidence"]
        )

    result["found"] = result["recommendation"] is not None or result["confidence"] is not None
    return result


def _normalize_recommendation(value: str) -> str | None:
    cleaned = re.sub(r"[^A-Z ]", "", value.upper()).strip()
    cleaned = re.sub(r"\s+", " ", cleaned)
    for label in VALID_RECOMMENDATIONS:
        if cleaned == label or cleaned.startswith(label):
            return label
    if "STRONG BUY" in cleaned or "OVERWEIGHT" in cleaned:
        return "STRONG BUY"
    if "BUY" in cleaned or "OUTPERFORM" in cleaned:
        return "BUY"
    if "SELL" in cleaned or "UNDERWEIGHT" in cleaned:
        return "SELL"
    if "HOLD" in cleaned or "NEUTRAL" in cleaned:
        return "HOLD"
    return None


def _parse_recommendation_from_prose(text: str) -> str | None:
    """Locate a recommendation label in Markdown prose.

    The recommendation section is searched first (a bold label is the expected
    format), then generic ``Recommendation: …`` lines. The disclaimer is stripped
    before the last-resort scan, otherwise phrases such as "a recommendation to
    buy or sell any security" would be misread as a BUY.
    """
    section = extract_markdown_section(
        text,
        ["recommendation", "final recommendation", "investment recommendation", "analyst verdict"],
    )
    scopes = [scope for scope in (section, text) if scope]

    bold_pattern = re.compile(
        r"\*\*\s*(STRONG\s+BUY|STRONG\s+SELL|BUY|SELL|HOLD)\s*\*\*", re.IGNORECASE
    )
    for scope in scopes:
        match = bold_pattern.search(scope)
        if match:
            label = _normalize_recommendation(match.group(1))
            if label:
                return label

    patterns = [
        r"\*{0,2}(?:final\s+)?recommendation\*{0,2}\s*(?:is|:|\-)\s*\*{0,2}([A-Za-z ]{3,20})",
        r"\*{0,2}(?:rating|action|verdict)\*{0,2}\s*(?:is|:|\-)\s*\*{0,2}([A-Za-z ]{3,20})",
    ]
    for scope in scopes:
        for pattern in patterns:
            match = re.search(pattern, scope, flags=re.IGNORECASE)
            if match:
                label = _normalize_recommendation(match.group(1))
                if label:
                    return label

    # Last resort: strongest label mentioned outside the disclaimer/methodology.
    if not section:
        return None
    for label in ("STRONG BUY", "STRONG SELL", "BUY", "SELL", "HOLD"):
        if re.search(rf"\b{label}\b", section.upper()):
            return label
    return None


def _clamp_confidence(value: Any) -> int | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        text = str(value)
        match = re.search(r"\d{1,3}", text)
        if not match:
            return None
        number = float(match.group())
    if number <= 1.0:  # 0-1 scale returned by some models
        number *= 100
    return int(max(0, min(100, round(number))))


_NUMBER = r"(\d{1,3}(?:\.\d+)?)"


def _parse_confidence_from_prose(text: str) -> int | None:
    """Find a confidence score in prose, accepting ``72``, ``0.72`` or ``72%``."""
    patterns = [
        rf"\*{{0,2}}confidence(?:\s+score)?\*{{0,2}}\s*[:\-]?\s*\*{{0,2}}\s*{_NUMBER}\s*%?",
        rf"confidence[^\n\d]{{0,20}}{_NUMBER}\s*(?:/\s*100|%)",
    ]
    for pattern in patterns:
        match = re.search(pattern, text, flags=re.IGNORECASE)
        if match:
            return _clamp_confidence(match.group(1))
    return None


def extract_markdown_section(text: str, headings: list[str], max_chars: int = 2000) -> str | None:
    """Return the body of the first matching Markdown heading.

    Args:
        text: Full Markdown document.
        headings: Candidate heading names, case-insensitive, without ``#``.
        max_chars: Maximum characters returned.
    """
    if not text:
        return None
    for heading in headings:
        pattern = re.compile(
            rf"^#{{1,4}}\s*{re.escape(heading)}\s*$([\s\S]*?)(?=^#{{1,4}}\s|\Z)",
            flags=re.IGNORECASE | re.MULTILINE,
        )
        match = pattern.search(text)
        if match and match.group(1).strip():
            return match.group(1).strip()[:max_chars]
    return None


def coerce_float(value: Any, default: float | None = None) -> float | None:
    """Best-effort float conversion (handles None, NaN, '12.3%', '1,234')."""
    if value is None:
        return default
    if isinstance(value, (int, float)) and value == value:  # reject NaN
        return float(value)
    cleaned = str(value).replace(",", "").replace("%", "").strip()
    try:
        return float(cleaned)
    except ValueError:
        return default


__all__ = [
    "InputValidationError",
    "VALID_RECOMMENDATIONS",
    "coerce_float",
    "extract_json_object",
    "extract_markdown_section",
    "normalize_ticker",
    "parse_recommendation",
    "validate_date_range",
]
