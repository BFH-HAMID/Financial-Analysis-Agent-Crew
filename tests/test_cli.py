"""Tests for the headless batch CLI (scripts/daily_report.py)."""

from __future__ import annotations

import json
from datetime import date

import pytest

from scripts import daily_report


class TestArgumentParsing:
    def test_defaults_from_environment(self, monkeypatch):
        monkeypatch.setenv("DAILY_TICKERS", "AAPL,MSFT")
        args = daily_report.parse_args([])
        assert args.tickers == "AAPL,MSFT"
        assert args.days == 180
        assert args.no_llm is False

    def test_explicit_dates_override_days(self):
        args = daily_report.parse_args(
            ["--tickers", "tsla", "--start", "2024-01-01", "--end", "2024-03-01"]
        )
        start, end = daily_report.resolve_dates(args)
        assert start == date(2024, 1, 1)
        assert end == date(2024, 3, 1)

    def test_days_window(self):
        args = daily_report.parse_args(["--days", "30"])
        start, end = daily_report.resolve_dates(args)
        assert (end - start).days >= 30


class TestRun:
    def test_summary_and_json_output(self, tmp_path, capsys):
        summary = tmp_path / "summary.json"
        exit_code = daily_report.main(
            [
                "--tickers",
                "AAPL",
                "--days",
                "60",
                "--no-llm",
                "--no-pdf",
                "--summary-json",
                str(summary),
            ]
        )
        assert exit_code == 0
        captured = capsys.readouterr().out
        assert "AAPL" in captured

        payload = json.loads(summary.read_text())
        assert payload["runs"][0]["ticker"] == "AAPL"
        assert payload["runs"][0]["is_demo"] is True
        assert payload["runs"][0]["recommendation"]
        assert payload["failures"] == {}

    def test_bad_ticker_is_reported_without_stopping_the_batch(self, tmp_path, capsys):
        summary = tmp_path / "summary.json"
        exit_code = daily_report.main(
            [
                "--tickers",
                "???,MSFT",
                "--days",
                "60",
                "--no-llm",
                "--no-pdf",
                "--summary-json",
                str(summary),
            ]
        )
        payload = json.loads(summary.read_text())
        # "???" fails validation, MSFT still produces a report.
        assert "???" in payload["failures"]
        assert [run["ticker"] for run in payload["runs"]] == ["MSFT"]
        assert exit_code == 0

    def test_no_tickers_returns_error(self):
        assert daily_report.main(["--tickers", " , ", "--no-llm"]) == 2

    def test_summarise_mentions_demo(self, analysis_result):
        line = daily_report.summarise(analysis_result)
        assert analysis_result.ticker in line
        assert "DEMO" in line

    @pytest.mark.parametrize("sender", ["send_telegram", "send_email"])
    def test_delivery_is_skipped_without_configuration(self, monkeypatch, sender):
        monkeypatch.delenv("TELEGRAM_BOT_TOKEN", raising=False)
        monkeypatch.delenv("TELEGRAM_CHAT_ID", raising=False)
        monkeypatch.delenv("SMTP_HOST", raising=False)
        if sender == "send_telegram":
            assert daily_report.send_telegram("hi") is False
        else:
            assert daily_report.send_email("subject", "body", []) is False
