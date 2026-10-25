"""Tests for logging, the conversation log, formatting and configuration."""

from __future__ import annotations

import json
import logging
from pathlib import Path

import pytest

from config.settings import Settings, get_settings
from utils.formatting import (
    fmt_compact,
    fmt_confidence,
    fmt_currency,
    fmt_number,
    fmt_pct,
    safe_float,
    sentiment_emoji,
)
from utils.logger import ConversationLog, get_logger


class TestConversationLog:
    def test_records_events(self):
        log = ConversationLog()
        log.agent_start("Researcher", "collecting news")
        log.tool_call("Researcher", "search_stock_news", "6 headlines")
        log.warning("Orchestrator", "demo mode")
        log.error("Writer", "boom")
        assert len(log) == 4
        assert [event.agent for event in log.events] == [
            "Researcher",
            "Researcher",
            "Orchestrator",
            "Writer",
        ]

    def test_durations_are_recorded(self):
        log = ConversationLog()
        log.agent_done("Analyst", "done", duration_s=1.234)
        assert log.events[0].duration_ms == pytest.approx(1234, abs=0.5)

    def test_markdown_rendering(self):
        log = ConversationLog()
        assert "No agent activity" in log.to_markdown()
        log.agent_start("Writer", "drafting")
        markdown = log.to_markdown()
        assert "Agent Conversation Log" in markdown
        assert "**Writer**" in markdown
        assert "drafting" in markdown

    def test_json_and_save(self, tmp_path):
        log = ConversationLog()
        log.agent_start("Writer", "drafting")
        payload = json.loads(log.to_json())
        assert payload["event_count"] == 1
        assert payload["events"][0]["agent"] == "Writer"

        path = log.save(tmp_path / "nested" / "log.json")
        assert path.exists()
        assert json.loads(path.read_text())["event_count"] == 1

    def test_max_events_trims_history(self):
        log = ConversationLog(max_events=10)
        for index in range(25):
            log.add("Agent", "tick", f"event {index}")
        assert len(log) == 10
        assert "event 24" in log.events[-1].message

    def test_timed_context_manager(self):
        log = ConversationLog()
        with log.timed("Analyst", message="working") as info:
            pass
        assert info["duration_s"] >= 0
        assert any(event.event == "finished" for event in log.events)


class TestLoggingSetup:
    def test_logger_is_namespaced_and_writes_file(self):
        logger = get_logger("financial_crew.tests")
        assert isinstance(logger, logging.Logger)
        logger.info("hello from the test suite")
        settings = get_settings()
        for handler in logging.getLogger("financial_crew").handlers:
            handler.flush()
        assert settings.log_file.exists()

    def test_root_has_console_and_file_handlers(self):
        get_logger("financial_crew")
        handlers = logging.getLogger("financial_crew").handlers
        assert len(handlers) >= 1


class TestFormatting:
    @pytest.mark.parametrize(
        "value,expected",
        [(12.345, "12.35"), (None, "n/a"), (float("nan"), "n/a"), ("7.1", "7.10")],
    )
    def test_fmt_number(self, value, expected):
        assert fmt_number(value) == expected

    def test_fmt_pct_signs(self):
        assert fmt_pct(12.5) == "+12.50%"
        assert fmt_pct(-3.2) == "-3.20%"
        assert fmt_pct(12.5, signed=False) == "12.50%"
        assert fmt_pct(None) == "n/a"

    def test_fmt_currency_and_compact(self):
        assert fmt_currency(1234.5) == "$1,234.50"
        assert fmt_compact(2_500_000_000) == "$2.50B"
        assert fmt_compact(1_234_567) == "$1.23M"
        assert fmt_compact(-4_500_000_000) == "-$4.50B"
        assert fmt_compact(None) == "n/a"

    def test_fmt_confidence(self):
        assert fmt_confidence(64) == "64/100"
        assert fmt_confidence(None) == "n/a"

    def test_safe_float(self):
        assert safe_float("3.5") == 3.5
        assert safe_float(float("inf")) is None
        assert safe_float(None) is None

    def test_sentiment_emoji(self):
        assert sentiment_emoji("Bullish") == "🟢"
        assert sentiment_emoji(None) == "⚪"


class TestSettings:
    def test_paths_exist(self):
        settings = get_settings()
        for path in (settings.reports_dir, settings.chart_dir, settings.pdf_dir, settings.log_dir):
            assert Path(path).exists()

    def test_public_dict_hides_secrets(self):
        public = get_settings().public_dict()
        assert "google_api_key" not in public
        assert "news_api_key" not in public
        assert public["app_name"]

    def test_env_overrides_are_read(self, monkeypatch):
        monkeypatch.setenv("GEMINI_MODEL", "gemini-test-model")
        monkeypatch.setenv("MAX_REVISION_ROUNDS", "5")
        monkeypatch.setenv("ALLOW_DEMO_DATA", "false")
        settings = Settings.from_env()
        assert settings.gemini_model == "gemini-test-model"
        assert settings.max_revision_rounds == 5
        assert settings.allow_demo_data is False

    def test_bad_numeric_env_falls_back(self, monkeypatch):
        monkeypatch.setenv("MAX_REVISION_ROUNDS", "not-a-number")
        assert Settings.from_env().max_revision_rounds == 2
