"""Tests for the sentiment engine."""

from __future__ import annotations

import json

import pytest

from tools.sentiment import (
    analyze_news_sentiment,
    analyze_texts,
    score_text,
    sentiment_label_emoji,
)


@pytest.mark.parametrize(
    "text",
    [
        "Company beats estimates with record profit and raises guidance",
        "Shares surge after strong demand and an analyst upgrade",
    ],
)
def test_positive_text_scores_positive(text):
    assert score_text(text) > 0.15


@pytest.mark.parametrize(
    "text",
    [
        "Company misses estimates and cuts guidance amid a profit warning",
        "Shares plunge after a probe and a wave of layoffs",
    ],
)
def test_negative_text_scores_negative(text):
    assert score_text(text) < -0.15


def test_neutral_text_is_neutral():
    assert abs(score_text("The company will report results next Tuesday.")) < 0.3


def test_empty_text_scores_zero():
    assert score_text("") == 0.0


def test_finance_lexicon_is_applied():
    # 'guidance cut' is not in VADER's lexicon; our overlay must catch it.
    assert score_text("The firm announced a guidance cut") < 0


def test_analyze_texts_aggregates(news):
    result = analyze_texts(news)
    assert result.articles_analyzed == len(news)
    assert result.positive_count + result.negative_count + result.neutral_count == len(news)
    assert -1 <= result.score <= 1
    assert result.label in {"bullish", "positive", "neutral", "negative", "bearish"}
    assert len(result.items) == len(news)
    assert all(-1 <= item["score"] <= 1 for item in result.items)


def test_analyze_texts_handles_empty_input():
    result = analyze_texts([])
    assert result.articles_analyzed == 0
    assert result.label == "neutral"
    assert result.score == 0.0
    assert any("No headlines" in note for note in result.observations)


def test_aggregate_label_thresholds():
    positive_items = [{"title": "Stock surges on record high and upgrade"} for _ in range(5)]
    negative_items = [{"title": "Stock plunges on bankruptcy and lawsuit"} for _ in range(5)]
    assert analyze_texts(positive_items).label in {"positive", "bullish"}
    assert analyze_texts(negative_items).label in {"negative", "bearish"}


def test_tool_wrapper_returns_json(news):
    payload = json.loads(analyze_news_sentiment(news))
    assert payload["articles_analyzed"] == len(news)
    assert "score" in payload and "label" in payload


def test_tool_wrapper_never_raises_on_bad_input():
    payload = json.loads(analyze_news_sentiment([{"title": None}, {"no_title": 1}]))
    assert "label" in payload


def test_to_dict_is_serialisable(news):
    data = analyze_texts(news).to_dict()
    json.dumps(data)  # must not raise
    assert {"score", "label", "items", "observations"}.issubset(data)


def test_emoji_mapping():
    assert sentiment_label_emoji("bullish") == "🟢"
    assert sentiment_label_emoji("bearish") == "🔴"
    assert sentiment_label_emoji("unknown-label") == "⚪"
