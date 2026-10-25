"""Tests for model construction and the live-crew degradation paths.

These tests never call the Gemini API: they model-construct and monkeypatch so
the whole suite runs offline in CI.
"""

from __future__ import annotations

from datetime import date
from types import SimpleNamespace

import pytest

from agents import llm as llm_module
from agents.llm import LLMConfigurationError, build_model, llm_is_configured
from agents.orchestrator import Orchestrator, response_text, validate_llm_draft


class TestModelFactory:
    def test_missing_key_raises(self, monkeypatch):
        fake_settings = SimpleNamespace(
            google_api_key=None,
            gemini_model="gemini-2.5-flash",
            gemini_temperature=0.3,
            llm_timeout_seconds=60,
            llm_available=False,
        )
        monkeypatch.setattr(llm_module, "get_settings", lambda: fake_settings)
        with pytest.raises(LLMConfigurationError):
            build_model()
        assert llm_is_configured() is False

    def test_model_is_built_with_configured_id(self, monkeypatch):
        fake_settings = SimpleNamespace(
            google_api_key="test-key",
            gemini_model="gemini-2.5-flash",
            gemini_temperature=0.3,
            llm_timeout_seconds=60,
            llm_available=True,
        )
        monkeypatch.setattr(llm_module, "get_settings", lambda: fake_settings)
        model = build_model(temperature=0.1)
        assert model.id == "gemini-2.5-flash"
        assert model.temperature == pytest.approx(0.1)
        assert llm_is_configured() is True


class TestResponseText:
    def test_plain_string(self):
        assert response_text("hello") == "hello"

    def test_object_with_content(self):
        class Fake:
            content = "## Report\nBody"

        assert response_text(Fake()) == "## Report\nBody"

    def test_list_content(self):
        class Fake:
            content = [{"text": "part one"}, "part two"]

        assert "part one" in response_text(Fake())
        assert "part two" in response_text(Fake())

    def test_none(self):
        assert response_text(None) == ""


class TestLiveCrewDegradation:
    def test_live_failure_falls_back_to_offline(self, monkeypatch):
        monkeypatch.setattr("agents.orchestrator.build_model", lambda *a, **k: object())

        def explode(self, **kwargs):
            raise RuntimeError("simulated upstream failure")

        monkeypatch.setattr(Orchestrator, "_run_live_crew", explode)
        orchestrator = Orchestrator(generate_pdf=False, max_revision_rounds=0)
        result = orchestrator.run("AAPL", date(2024, 1, 1), date(2024, 5, 1))

        assert result.mode == "offline"
        assert result.llm_used is False
        assert result.report_markdown
        assert result.recommendation is not None
        assert any("Live agent crew failed" in warning for warning in result.warnings)

    def test_llm_draft_is_used_and_gaps_are_filled(self, monkeypatch):
        monkeypatch.setattr("agents.orchestrator.build_model", lambda *a, **k: object())

        # Three of the five core sections are present, so the draft is trusted;
        # the missing confidence score and disclaimer must be appended.
        draft = (
            "# AAPL — Financial Analysis Report\n\n"
            "## Executive Summary\nA concise summary citing 12.5% growth and 21.4x P/E, "
            "3.8% dividend growth and 1.9 beta-adjusted returns over the window.\n\n"
            "## KPI Table\n| Metric | Value |\n| --- | --- |\n| P/E | 21.4 |\n"
            "| ROI | 12.5% |\n| RSI | 58.2 |\n\n"
            "## Recommendation\n**BUY** — momentum is positive and valuation is fair.\n"
        )

        def fake_crew(self, **kwargs):
            return ("research brief", "analyst brief", draft, 1)

        monkeypatch.setattr(Orchestrator, "_run_live_crew", fake_crew)
        orchestrator = Orchestrator(generate_pdf=False)
        result = orchestrator.run("AAPL", date(2024, 1, 1), date(2024, 5, 1))

        assert result.mode == "live"
        assert result.llm_used is True
        assert result.revision_rounds == 1
        assert result.recommendation == "BUY"
        # Missing sections are appended by the deterministic fallback.
        assert "Appendix" in result.report_markdown
        assert "Confidence Score" in result.report_markdown
        assert "Disclaimer" in result.report_markdown
        # Demo runs must disclose synthetic data even if the LLM forgot.
        assert "SYNTHETIC DEMO DATA" in result.report_markdown.upper()
        assert result.confidence is not None and result.confidence <= 65

    def test_llm_confidence_is_parsed(self, monkeypatch):
        monkeypatch.setattr("agents.orchestrator.build_model", lambda *a, **k: object())
        draft = (
            "# AAPL — Financial Analysis Report\n"
            "## Executive Summary\nSolid quarter with 12.5% revenue growth.\n"
            "## KPI Table\n| Metric | Value |\n| --- | --- |\n| P/E | 21.4 |\n"
            "## Recommendation\n**HOLD** — valuation is full.\n"
            "## Confidence Score\n**Confidence:** 58/100\n"
            "## Disclaimer\nEducational use only.\n"
            "> SYNTHETIC DEMO DATA\n"
        )
        monkeypatch.setattr(
            Orchestrator,
            "_run_live_crew",
            lambda self, **kwargs: ("r", "a", draft, 0),
        )
        result = Orchestrator(generate_pdf=False).run("AAPL", date(2024, 1, 1), date(2024, 5, 1))
        assert result.recommendation == "HOLD"
        assert result.confidence == 58


class TestDraftValidation:
    GOOD_DRAFT = (
        "# AAPL — Financial Analysis Report\n\n"
        "## Executive Summary\nRevenue grew 12.5% while margins improved 1.4pp.\n\n"
        "## KPI Table\n| Metric | Value |\n| --- | --- |\n| P/E | 21.4 |\n\n"
        "## Recommendation\n**BUY** — momentum and valuation are supportive.\n\n"
        "## Confidence Score\n**Confidence:** 62/100\n\n"
        "## Disclaimer\nEducational use only, not investment advice.\n"
        "## Methodology & Limitations\nDaily closes for the selected window, "
        "annualised with 252 trading days.\n"
    )

    def test_good_draft_passes(self):
        assert validate_llm_draft(self.GOOD_DRAFT) is None

    @pytest.mark.parametrize(
        "marker",
        [
            "Error in Team run: boom",
            "Model provider error after 2 attempts",
            "Unknown error from Gemini API: TLS",
            "API key not valid. Please pass a valid API key.",
        ],
    )
    def test_provider_errors_are_rejected(self, marker):
        draft = self.GOOD_DRAFT + f"\n> {marker}\n"
        reason = validate_llm_draft(draft)
        assert reason is not None
        assert "provider error" in reason

    def test_short_output_is_rejected(self):
        assert validate_llm_draft("Too short.") is not None

    def test_missing_structure_is_rejected(self):
        draft = "## Notes\n" + "Some prose without headings. " * 40
        assert validate_llm_draft(draft) is not None

    def test_unusable_live_output_falls_back_to_offline(self, monkeypatch):
        monkeypatch.setattr("agents.orchestrator.build_model", lambda *a, **k: object())
        bad_draft = "Error in Team run: TLS/SSL connection has been closed"
        monkeypatch.setattr(
            Orchestrator, "_run_live_crew", lambda self, **kwargs: ("r", "a", bad_draft, 0)
        )
        result = Orchestrator(generate_pdf=False).run("AAPL", date(2024, 1, 1), date(2024, 5, 1))
        assert result.mode == "offline"
        assert result.llm_used is False
        assert result.recommendation is not None
