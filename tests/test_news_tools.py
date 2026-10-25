"""Tests for news retrieval, including provider failure handling."""

from __future__ import annotations

import json

import pytest

from tools import news_tools
from tools.news_tools import NewsError, build_news_query, search_news, search_stock_news


def _boom(*args, **kwargs):
    raise RuntimeError("provider unreachable")


def test_demo_news_is_used_when_providers_fail(monkeypatch):
    monkeypatch.setattr(news_tools, "_search_duckduckgo_news", _boom)
    monkeypatch.setattr(news_tools, "_search_duckduckgo_text", _boom)
    items = search_news("AAPL", "Apple Inc.", max_results=4)
    assert len(items) == 4
    assert all(item["is_demo"] for item in items)
    assert all(item["title"] for item in items)


def test_search_news_raises_when_demo_disabled(monkeypatch):
    monkeypatch.setattr(news_tools, "_search_duckduckgo_news", _boom)
    monkeypatch.setattr(news_tools, "_search_duckduckgo_text", _boom)
    with pytest.raises(NewsError):
        search_news("AAPL", max_results=3, allow_demo=False)


def _today_iso(days_ago: int = 0) -> str:
    from datetime import date, timedelta

    return (date.today() - timedelta(days=days_ago)).isoformat()


def test_successful_provider_output_is_cleaned(monkeypatch):
    today = _today_iso()
    fake = [
        {
            "title": "Apple unveils new product lineup for the holiday quarter",
            "url": "https://example.com/a",
            "source": "Example",
            "published": today,
            "summary": "Details inside.",
        },
        {  # duplicate title -> dropped
            "title": "Apple unveils new product lineup for the holiday quarter",
            "url": "https://example.com/b",
            "source": "Other",
            "published": today,
            "summary": "",
        },
        {  # too short -> dropped
            "title": "Short",
            "url": "",
            "source": "x",
            "published": today,
        },
    ]
    monkeypatch.setattr(news_tools, "_search_duckduckgo_news", lambda *a, **k: fake)
    items = search_news("AAPL", max_results=5)
    assert len(items) == 1
    assert items[0]["is_demo"] is False
    assert items[0]["source"] == "Example"


def test_results_are_sorted_newest_first(monkeypatch):
    fake = [
        {
            "title": "Older headline about the company results",
            "published": _today_iso(9),
            "source": "A",
        },
        {
            "title": "Newest headline about the company results",
            "published": _today_iso(0),
            "source": "B",
        },
        {
            "title": "Middle headline about the company results",
            "published": _today_iso(4),
            "source": "C",
        },
    ]
    monkeypatch.setattr(news_tools, "_search_duckduckgo_news", lambda *a, **k: fake)
    items = search_news("AAPL", max_results=5)
    dates = [item["published"] for item in items]
    assert dates == sorted(dates, reverse=True)


def test_max_results_is_respected(monkeypatch):
    monkeypatch.setattr(news_tools, "_search_duckduckgo_news", _boom)
    monkeypatch.setattr(news_tools, "_search_duckduckgo_text", _boom)
    assert len(search_news("TSLA", max_results=2)) <= 2


def test_stock_news_tool_returns_json(monkeypatch):
    monkeypatch.setattr(news_tools, "_search_duckduckgo_news", _boom)
    monkeypatch.setattr(news_tools, "_search_duckduckgo_text", _boom)
    payload = json.loads(search_stock_news("AAPL", "Apple Inc.", 3))
    assert payload["count"] == 3
    assert payload["all_synthetic"] is True
    assert "SYNTHETIC" in payload["notice"].upper()


def test_stock_news_tool_clamps_result_count(monkeypatch):
    monkeypatch.setattr(news_tools, "_search_duckduckgo_news", _boom)
    monkeypatch.setattr(news_tools, "_search_duckduckgo_text", _boom)
    payload = json.loads(search_stock_news("AAPL", "", 999))
    assert payload["count"] <= 15


def test_build_news_query():
    assert "Apple Inc." in build_news_query("AAPL", "Apple Inc.")
    assert "AAPL" in build_news_query("AAPL")


def test_parse_date_handles_common_formats():
    assert str(news_tools._parse_date("2024-06-01T12:30:00Z")) == "2024-06-01"
    assert str(news_tools._parse_date("2024-06-01")) == "2024-06-01"
    assert news_tools._parse_date("not a date") is None
    assert news_tools._parse_date(None) is None
