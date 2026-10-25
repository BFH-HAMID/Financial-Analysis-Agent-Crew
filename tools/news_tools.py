"""News retrieval for the Market Researcher agent.

Provider order (first success wins):

1. DuckDuckGo news search (``ddgs``, no API key required).
2. DuckDuckGo web/text search (no key).
3. NewsAPI.org (only when ``NEWS_API_KEY`` is configured).
4. Synthetic demo headlines, clearly labelled, when ``ALLOW_DEMO_DATA`` is on.

Every provider call is time-boxed and wrapped in error handling so a report can
still be produced when news is unavailable.
"""

from __future__ import annotations

import json
from datetime import date, datetime, timedelta
from typing import Any

import requests

from config.settings import get_settings
from tools.demo_data import DEMO_NOTICE, demo_news
from utils.logger import get_logger

logger = get_logger(__name__)

NEWSAPI_ENDPOINT = "https://newsapi.org/v2/everything"
_USER_AGENT = "Mozilla/5.0 (compatible; FinancialAnalysisAgentCrew/1.0)"


class NewsError(RuntimeError):
    """Raised when no news provider can return headlines."""


# ---------------------------------------------------------------------------
# Provider implementations
# ---------------------------------------------------------------------------
def _search_duckduckgo_news(query: str, max_results: int, timeout: int) -> list[dict[str, Any]]:
    """DuckDuckGo news results via the ddgs package."""
    from ddgs import DDGS

    results: list[dict[str, Any]] = []
    with DDGS(timeout=timeout) as client:
        for item in client.news(query, max_results=max_results):
            results.append(
                {
                    "title": item.get("title") or "",
                    "url": item.get("url") or item.get("href") or "",
                    "source": item.get("source") or "DuckDuckGo",
                    "published": item.get("date") or "",
                    "summary": item.get("body") or "",
                    "query": query,
                }
            )
    return results


def _search_duckduckgo_text(query: str, max_results: int, timeout: int) -> list[dict[str, Any]]:
    """DuckDuckGo web results, used when the news endpoint is unavailable."""
    from ddgs import DDGS

    results: list[dict[str, Any]] = []
    with DDGS(timeout=timeout) as client:
        for item in client.text(f"{query} stock news", max_results=max_results):
            results.append(
                {
                    "title": item.get("title") or "",
                    "url": item.get("href") or item.get("url") or "",
                    "source": item.get("source") or "DuckDuckGo",
                    "published": "",
                    "summary": item.get("body") or "",
                    "query": query,
                }
            )
    return results


def _search_newsapi(
    query: str, max_results: int, timeout: int, api_key: str
) -> list[dict[str, Any]]:
    """NewsAPI.org fallback (requires ``NEWS_API_KEY``)."""
    params = {
        "q": query,
        "language": "en",
        "sortBy": "publishedAt",
        "pageSize": min(max_results, 100),
    }
    response = requests.get(
        NEWSAPI_ENDPOINT,
        params=params,
        headers={"X-Api-Key": api_key, "User-Agent": _USER_AGENT},
        timeout=timeout,
    )
    response.raise_for_status()
    payload = response.json()
    if payload.get("status") != "ok":
        raise NewsError(payload.get("message") or "NewsAPI returned a non-ok status")

    return [
        {
            "title": article.get("title") or "",
            "url": article.get("url") or "",
            "source": (article.get("source") or {}).get("name") or "NewsAPI",
            "published": (article.get("publishedAt") or "")[:10],
            "summary": article.get("description") or "",
            "query": query,
        }
        for article in payload.get("articles", [])
    ]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
def search_news(
    ticker: str,
    company_name: str = "",
    days: int | None = None,
    max_results: int | None = None,
    *,
    allow_demo: bool | None = None,
) -> list[dict[str, Any]]:
    """Collect recent news headlines about a company.

    Args:
        ticker: Stock ticker, e.g. ``"AAPL"``.
        company_name: Optional company name to broaden the query.
        days: Look-back window in days (defaults to settings).
        max_results: Maximum headlines to return.
        allow_demo: Override the synthetic fallback switch.

    Returns:
        A list of dicts with ``title``, ``url``, ``source``, ``published`` and
        ``summary``.

    Raises:
        NewsError: If every provider fails and demo fallback is disabled.
    """
    settings = get_settings()
    days = days or settings.news_lookback_days
    max_results = max_results or settings.max_news_results
    allow_demo = settings.allow_demo_data if allow_demo is None else allow_demo
    timeout = settings.request_timeout_seconds

    primary_query = f"{company_name or ticker} stock news"
    if company_name:
        primary_query = f"{company_name} ({ticker}) stock news"

    failures: list[str] = []
    providers = [
        ("duckduckgo-news", lambda: _search_duckduckgo_news(primary_query, max_results, timeout)),
        ("duckduckgo-web", lambda: _search_duckduckgo_text(ticker, max_results, timeout)),
    ]
    if settings.news_api_key:
        providers.append(
            (
                "newsapi",
                lambda: _search_newsapi(
                    f"{company_name or ticker}", max_results, timeout, settings.news_api_key or ""
                ),
            )
        )

    for name, call in providers:
        try:
            items = call()
        except Exception as exc:  # noqa: BLE001 - any provider failure must be survivable
            failures.append(f"{name}: {type(exc).__name__}: {exc}")
            logger.warning("News provider %s failed: %s", name, exc)
            continue

        cleaned = _clean_items(items, max_results, days)
        if cleaned:
            logger.info("Retrieved %s news items for %s via %s", len(cleaned), ticker, name)
            return cleaned
        failures.append(f"{name}: no usable results")

    if allow_demo:
        logger.warning("All news providers failed for %s — using demo headlines", ticker)
        items = demo_news(ticker, company_name, max_results)
        for item in items:
            item["provider"] = "synthetic-demo"
        return items

    raise NewsError(
        "No news could be retrieved ("
        + "; ".join(failures)
        + "). Set NEWS_API_KEY or enable ALLOW_DEMO_DATA=true for a demo run."
    )


def _clean_items(items: list[dict[str, Any]], max_results: int, days: int) -> list[dict[str, Any]]:
    """De-duplicate, filter by date window and date-sort news items."""
    cutoff = date.today() - timedelta(days=max(days, 1))
    seen_titles: set[str] = set()
    cleaned: list[dict[str, Any]] = []

    for item in items:
        title = (item.get("title") or "").strip()
        if len(title) < 12:
            continue
        key = title.lower()[:80]
        if key in seen_titles:
            continue

        published = _parse_date(item.get("published"))
        if published is not None and published < cutoff:
            continue
        seen_titles.add(key)
        cleaned.append(
            {
                "title": title,
                "url": item.get("url") or "",
                "source": item.get("source") or "unknown",
                "published": published.isoformat() if published else "",
                "summary": (item.get("summary") or "").strip()[:400],
                "query": item.get("query") or "",
                "is_demo": False,
            }
        )

    cleaned.sort(key=lambda entry: entry.get("published") or "", reverse=True)
    return cleaned[:max_results]


def _parse_date(raw: Any) -> date | None:
    """Parse the various date formats news providers return."""
    if not raw:
        return None
    text = str(raw).strip()

    for fmt in ("%Y-%m-%dT%H:%M:%SZ", "%Y-%m-%d %H:%M:%S", "%Y-%m-%d", "%Y/%m/%d"):
        try:
            return datetime.strptime(text[: len(fmt) + 2], fmt).date()
        except ValueError:
            continue

    try:  # NewsAPI ISO timestamps
        return datetime.fromisoformat(text.replace("Z", "+00:00")).date()
    except ValueError:
        return None


def search_stock_news(ticker: str, company_name: str = "", max_results: int = 8) -> str:
    """Search recent news headlines about a stock and return them as JSON.

    Use this tool to gather qualitative, up-to-date context for a company.

    Args:
        ticker: Stock ticker symbol, e.g. "AAPL".
        company_name: Optional company name to improve search relevance.
        max_results: Maximum number of headlines to return (1-15).

    Returns:
        A JSON string containing a list of headlines with source and date.
    """
    try:
        limit = max(1, min(int(max_results or 8), 15))
        items = search_news(ticker=ticker, company_name=company_name, max_results=limit)
        return json.dumps(
            {
                "ticker": ticker,
                "count": len(items),
                "items": items,
                "all_synthetic": all(item.get("is_demo") for item in items),
                "notice": DEMO_NOTICE if any(item.get("is_demo") for item in items) else "",
            },
            default=str,
        )
    except Exception as exc:  # noqa: BLE001
        logger.exception("search_stock_news failed for %s", ticker)
        return json.dumps({"error": f"{type(exc).__name__}: {exc}", "ticker": ticker, "items": []})


def build_news_query(ticker: str, company_name: str = "") -> str:
    """Build a human-readable search query string (used in the UI and log)."""
    if company_name:
        return f"{company_name} ({ticker}) stock news"
    return f"{ticker} stock news"


__all__ = [
    "NewsError",
    "build_news_query",
    "search_news",
    "search_stock_news",
]
