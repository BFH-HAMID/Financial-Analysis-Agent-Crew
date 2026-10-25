"""Sentiment analysis for news headlines and summaries.

The default backend is a deterministic lexicon model (VADER) with a finance
vocabulary overlay, which needs no API key and is fast. Setting
``SENTIMENT_BACKEND=llm`` switches the aggregate label to an LLM-judged value
(handled by the Market Researcher agent prompt) while the per-item scores remain
lexicon-based, so the report always has numbers to show.
"""

from __future__ import annotations

import re
from collections.abc import Iterable
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

from config.settings import get_settings
from utils.logger import get_logger

logger = get_logger(__name__)

#: Extra finance terms VADER does not know about. Value in the range -4 … +4.
FINANCE_LEXICON: dict[str, float] = {
    "beat": 2.2,
    "beats": 2.2,
    "missed": -2.2,
    "misses": -2.2,
    "upgrade": 2.4,
    "upgraded": 2.4,
    "downgrade": -2.4,
    "downgraded": -2.4,
    "outperform": 2.0,
    "underperform": -2.0,
    "overweight": 1.6,
    "underweight": -1.6,
    "bullish": 2.2,
    "bearish": -2.2,
    "record high": 2.5,
    "all-time high": 2.5,
    "rally": 2.0,
    "selloff": -2.2,
    "sell-off": -2.2,
    "plunge": -2.8,
    "plunges": -2.8,
    "surge": 2.4,
    "surges": 2.4,
    "soar": 2.4,
    "soars": 2.4,
    "tumble": -2.4,
    "tumbles": -2.4,
    "slump": -2.2,
    "layoffs": -2.2,
    "lawsuit": -2.0,
    "probe": -1.8,
    "investigation": -1.8,
    "recall": -2.0,
    "bankruptcy": -3.2,
    "default": -2.4,
    "dividend hike": 1.8,
    "buyback": 1.6,
    "guidance cut": -2.4,
    "guidance raise": 2.2,
    "profit warning": -2.6,
    "strong demand": 2.0,
    "weak demand": -2.0,
    "supply chain": -0.6,
}

_NEGATION_TOKENS = {"not", "no", "never", "without", "isn't", "wasn't", "aren't", "doesn't"}

_POSITIVE_THRESHOLD = 0.15
_NEGATIVE_THRESHOLD = -0.15


@dataclass
class SentimentResult:
    """Aggregated sentiment for a set of text items."""

    score: float = 0.0
    label: str = "neutral"
    positive_count: int = 0
    negative_count: int = 0
    neutral_count: int = 0
    articles_analyzed: int = 0
    backend: str = "lexicon"
    observations: list[str] = field(default_factory=list)
    items: list[dict[str, Any]] = field(default_factory=list)
    generated_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat(timespec="seconds")
    )

    def to_dict(self) -> dict[str, Any]:
        return {
            "score": round(self.score, 3),
            "label": self.label,
            "positive_count": self.positive_count,
            "negative_count": self.negative_count,
            "neutral_count": self.neutral_count,
            "articles_analyzed": self.articles_analyzed,
            "backend": self.backend,
            "observations": self.observations,
            "items": self.items,
            "generated_at": self.generated_at,
        }


def _get_analyzer():
    """Build (once) the VADER analyzer with the finance lexicon merged in."""
    global _ANALYZER
    try:
        return _ANALYZER  # type: ignore[name-defined]
    except NameError:
        pass

    try:
        from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

        analyzer = SentimentIntensityAnalyzer()
        analyzer.lexicon.update(FINANCE_LEXICON)
    except Exception as exc:  # pragma: no cover - vaderSentiment is a dependency
        logger.warning("VADER unavailable (%s); using a simple keyword counter", exc)
        analyzer = None
    _ANALYZER = analyzer  # type: ignore[name-defined]
    return analyzer


def score_text(text: str) -> float:
    """Return a compound sentiment score in ``[-1, 1]`` for a single text."""
    if not text:
        return 0.0

    analyzer = _get_analyzer()
    if analyzer is not None:
        score = float(analyzer.polarity_scores(text)["compound"])
    else:
        lowered = text.lower()
        score = sum(value for term, value in FINANCE_LEXICON.items() if term in lowered) / 10

    # Light negation handling for keyword matches VADER may not catch.
    words = re.findall(r"[a-z']+", text.lower())
    if any(token in _NEGATION_TOKENS for token in words):
        score *= 0.6

    return max(-1.0, min(1.0, score))


def _classify(score: float) -> str:
    if score >= _POSITIVE_THRESHOLD:
        return "positive"
    if score <= _NEGATIVE_THRESHOLD:
        return "negative"
    return "neutral"


def analyze_texts(items: Iterable[dict[str, Any]]) -> SentimentResult:
    """Score a list of news items and aggregate the result.

    Args:
        items: Dicts with at least ``title``; optional ``summary`` and ``source``.

    Returns:
        A :class:`SentimentResult` with per-item scores and observations.
    """
    items = list(items or [])
    settings = get_settings()

    scored: list[dict[str, Any]] = []
    counts = {"positive": 0, "negative": 0, "neutral": 0}

    for item in items:
        title = str(item.get("title") or "").strip()
        summary = str(item.get("summary") or "").strip()
        # Headlines carry most of the signal; summaries get a lower weight.
        title_score = score_text(title)
        summary_score = score_text(summary) if summary else 0.0
        score = title_score * 0.7 + summary_score * 0.3
        label = _classify(score)
        counts[label] += 1
        scored.append(
            {
                "title": title,
                "source": item.get("source") or "unknown",
                "published": item.get("published") or "",
                "url": item.get("url") or "",
                "score": round(score, 3),
                "label": label,
                "is_demo": bool(item.get("is_demo")),
            }
        )

    total = len(scored)
    average = sum(entry["score"] for entry in scored) / total if total else 0.0

    if total == 0:
        label = "neutral"
    elif average >= 0.25:
        label = "bullish"
    elif average >= _POSITIVE_THRESHOLD:
        label = "positive"
    elif average <= -0.25:
        label = "bearish"
    elif average <= _NEGATIVE_THRESHOLD:
        label = "negative"
    else:
        label = "neutral"

    observations: list[str] = []
    if total == 0:
        observations.append("No headlines were available, so sentiment defaults to neutral.")
    if (
        counts["positive"]
        and counts["negative"]
        and abs(counts["positive"] - counts["negative"]) <= 2
    ):
        observations.append("Coverage is mixed: positive and negative headlines roughly balance.")
    if total and counts["positive"] / total >= 0.6:
        observations.append("Headline tone is predominantly positive.")
    if total and counts["negative"] / total >= 0.6:
        observations.append("Headline tone is predominantly negative.")
    strongly_negative = [entry for entry in scored if entry["score"] <= -0.5]
    if strongly_negative:
        observations.append(
            f"{len(strongly_negative)} strongly negative headline(s) detected — "
            "review them for company-specific risks."
        )
    if any(entry.get("is_demo") for entry in scored):
        observations.append("Sentiment is computed on synthetic demo headlines.")

    return SentimentResult(
        score=round(average, 3),
        label=label,
        positive_count=counts["positive"],
        negative_count=counts["negative"],
        neutral_count=counts["neutral"],
        articles_analyzed=total,
        backend=settings.sentiment_backend,
        observations=observations,
        items=scored,
    )


def analyze_news_sentiment(items: Iterable[dict[str, Any]]) -> str:
    """Aggregate sentiment across news items.

    Use this tool after collecting headlines to quantify market mood.

    Args:
        items: List of news dicts, each with a ``title`` (and optional ``summary``).

    Returns:
        A JSON string with the aggregate score, label and per-headline scores.
    """
    import json

    try:
        return json.dumps(analyze_texts(items).to_dict(), default=str)
    except Exception as exc:  # noqa: BLE001
        logger.exception("Sentiment analysis failed")
        return json.dumps(
            {"error": f"{type(exc).__name__}: {exc}", "label": "neutral", "score": 0.0}
        )


def sentiment_label_emoji(label: str) -> str:
    """Return an emoji for a sentiment label."""
    return {
        "bullish": "🟢",
        "positive": "🟢",
        "neutral": "🟡",
        "negative": "🔴",
        "bearish": "🔴",
    }.get((label or "").lower(), "⚪")


__all__ = [
    "FINANCE_LEXICON",
    "SentimentResult",
    "analyze_news_sentiment",
    "analyze_texts",
    "score_text",
    "sentiment_label_emoji",
]
