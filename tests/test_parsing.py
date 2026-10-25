"""Tests for input validation and LLM output parsing."""

from __future__ import annotations

import json
from datetime import date, timedelta

import pytest

from utils.parsing import (
    InputValidationError,
    coerce_float,
    extract_json_object,
    extract_markdown_section,
    normalize_ticker,
    parse_recommendation,
    validate_date_range,
)


class TestTicker:
    @pytest.mark.parametrize(
        "raw,expected",
        [
            ("aapl", "AAPL"),
            (" aapl ", "AAPL"),
            ("brk-b", "BRK-B"),
            ("0700.hk", "0700.HK"),
            ("^gspc", "^GSPC"),
        ],
    )
    def test_normalize(self, raw, expected):
        assert normalize_ticker(raw) == expected

    @pytest.mark.parametrize("raw", ["", "   ", "AA PL?X", "!!!", "A" * 30, "^", " "])
    def test_invalid(self, raw):
        with pytest.raises(InputValidationError):
            normalize_ticker(raw)


class TestDates:
    def test_valid_range(self):
        start, end = validate_date_range("2024-01-01", "2024-06-01")
        assert (end - start).days > 30
        assert isinstance(start, date)

    def test_reversed_range_is_rejected(self):
        with pytest.raises(InputValidationError):
            validate_date_range("2024-06-01", "2024-01-01")

    def test_too_short_range_is_rejected(self):
        with pytest.raises(InputValidationError):
            validate_date_range("2024-01-01", "2024-01-03")

    def test_future_end_is_clamped(self):
        future = date.today() + timedelta(days=30)
        start = date.today() - timedelta(days=60)
        _, end = validate_date_range(start, future)
        assert end <= date.today()

    def test_bad_string_is_rejected(self):
        with pytest.raises(InputValidationError):
            validate_date_range("01/01/2024", "2024-06-01")


class TestJsonExtraction:
    def test_plain_object(self):
        assert extract_json_object('{"a": 1}') == {"a": 1}

    def test_fenced_block(self):
        text = 'Here you go:\n```json\n{"approved": false, "feedback": "add data"}\n```\nThanks'
        assert extract_json_object(text)["feedback"] == "add data"

    def test_surrounded_by_prose(self):
        text = 'Result: {"recommendation": "BUY", "confidence": 72} — done.'
        assert extract_json_object(text)["recommendation"] == "BUY"

    def test_nested_object(self):
        text = '{"outer": {"inner": [1, 2]}, "ok": true}'
        assert extract_json_object(text)["outer"]["inner"] == [1, 2]

    def test_no_json(self):
        assert extract_json_object("no json here") is None
        assert extract_json_object("") is None


class TestRecommendationParsing:
    def test_from_json(self):
        parsed = parse_recommendation('{"recommendation": "Strong Buy", "confidence": 78}')
        assert parsed["recommendation"] == "STRONG BUY"
        assert parsed["confidence"] == 78
        assert parsed["found"] is True

    def test_from_bold_markdown(self):
        report = (
            "## Recommendation\n**BUY** — momentum is positive and valuation is reasonable.\n\n"
            "## Confidence Score\n**Confidence:** 64/100\n"
        )
        parsed = parse_recommendation(report)
        assert parsed["recommendation"] == "BUY"
        assert parsed["confidence"] == 64

    def test_hold_wins_over_disclaimer_word_buy(self):
        report = (
            "## Recommendation\n**HOLD** — valuation is full.\n\n"
            "## Disclaimer\nThis is not a recommendation to buy or sell any security.\n"
        )
        assert parse_recommendation(report)["recommendation"] == "HOLD"

    def test_prose_recommendation_line(self):
        report = "### Recommendation\nThe recommendation is SELL given the drawdown."
        assert parse_recommendation(report)["recommendation"] == "SELL"

    def test_confidence_fraction_is_scaled(self):
        report = "## Recommendation\n**BUY**\n## Confidence Score\nConfidence: 0.72"
        assert parse_recommendation(report)["confidence"] == 72

    def test_confidence_is_clamped(self):
        assert parse_recommendation('{"confidence": 140}')["confidence"] == 100
        assert parse_recommendation('{"confidence": -5}')["confidence"] == 0

    def test_missing_everything(self):
        parsed = parse_recommendation("Nothing useful here.")
        assert parsed["found"] is False
        assert parsed["recommendation"] is None
        assert parsed["confidence"] is None

    def test_empty_input(self):
        assert parse_recommendation("")["found"] is False


class TestMarkdownSection:
    def test_extracts_body(self):
        text = "# Title\n\n## Executive Summary\nGood quarter.\n\n## KPI Table\n| a | b |"
        assert extract_markdown_section(text, ["executive summary"]) == "Good quarter."

    def test_case_insensitive_and_missing(self):
        text = "## RISK ASSESSMENT\nHigh volatility."
        assert "High volatility" in extract_markdown_section(text, ["risk assessment"])
        assert extract_markdown_section(text, ["recommendation"]) is None


def test_coerce_float():
    assert coerce_float("1,234.5") == 1234.5
    assert coerce_float("12.3%") == 12.3
    assert coerce_float(None) is None
    assert coerce_float("abc", default=0.0) == 0.0


def test_parse_recommendation_output_is_json_friendly():
    json.dumps(parse_recommendation("**SELL** — high leverage"))
