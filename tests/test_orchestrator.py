"""End-to-end tests for the orchestrator, offline crew and report QA."""

from __future__ import annotations

import json
from datetime import date

import pytest

from agents.offline_crew import (
    analyst_brief,
    compose_report,
    heuristic_recommendation,
    research_brief,
)
from agents.orchestrator import (
    AnalysisResult,
    Orchestrator,
    ensure_report_sections,
    has_section,
    review_draft,
    split_sections,
)
from utils.parsing import VALID_RECOMMENDATIONS, InputValidationError


class TestAnalysisRun:
    def test_run_produces_complete_result(self, analysis_result):
        result = analysis_result
        assert isinstance(result, AnalysisResult)
        assert result.ticker == "AAPL"
        assert result.mode == "offline"
        assert result.llm_used is False
        assert len(result.report_markdown) > 2000
        assert result.recommendation in VALID_RECOMMENDATIONS
        assert 0 <= result.confidence <= 100
        assert result.is_demo is True
        assert len(result.news) > 0
        assert result.sentiment["articles_analyzed"] == len(result.news)
        assert set(result.chart_images) == {"price", "volume", "rsi", "kpi"}

    def test_report_contains_every_required_section(self, analysis_result):
        for key in (
            "executive_summary",
            "company_snapshot",
            "news_sentiment",
            "kpi_table",
            "charts",
            "risk_assessment",
            "recommendation",
            "confidence",
            "methodology",
            "disclaimer",
        ):
            assert has_section(analysis_result.report_markdown, key), f"missing section: {key}"

    def test_report_discloses_synthetic_data(self, analysis_result):
        assert "SYNTHETIC DEMO DATA" in analysis_result.report_markdown.upper()

    def test_conversation_log_is_populated(self, analysis_result):
        events = analysis_result.conversation
        assert len(events) >= 8
        agents = {event["agent"] for event in events}
        assert "Orchestrator" in agents
        assert any("Analyst" in agent for agent in agents)
        assert any("Researcher" in agent for agent in agents)
        json.dumps(events)  # must be serialisable

    def test_artefacts_are_written(self, analysis_result, project_root):
        assert analysis_result.report_path and analysis_result.log_path
        assert analysis_result.pdf_path
        from pathlib import Path

        for path in (
            analysis_result.report_path,
            analysis_result.log_path,
            analysis_result.pdf_path,
        ):
            assert Path(path).exists()
            assert Path(path).stat().st_size > 500
        assert str(project_root) in analysis_result.report_path

    def test_duration_and_timings(self, analysis_result):
        assert analysis_result.duration_s > 0
        assert "market_data" in analysis_result.timings

    def test_confidence_is_capped_for_demo_data(self, analysis_result):
        assert analysis_result.confidence <= 65

    def test_invalid_ticker_raises(self):
        orchestrator = Orchestrator(allow_llm=False, generate_pdf=False)
        with pytest.raises(InputValidationError):
            orchestrator.run("???", date(2024, 1, 1), date(2024, 6, 1))

    def test_progress_callback_receives_updates(self):
        updates: list[tuple[str, float, str]] = []
        orchestrator = Orchestrator(
            allow_llm=False,
            generate_pdf=False,
            progress=lambda stage, fraction, message: updates.append((stage, fraction, message)),
        )
        orchestrator.run("MSFT", date(2024, 1, 1), date(2024, 5, 1))
        assert len(updates) >= 5
        assert updates[-1][0] == "Done"
        assert updates[-1][1] == pytest.approx(1.0)

    def test_max_revision_rounds_zero_is_respected(self):
        orchestrator = Orchestrator(allow_llm=False, generate_pdf=False, max_revision_rounds=0)
        result = orchestrator.run("TSLA", date(2024, 1, 1), date(2024, 4, 1))
        assert result.revision_rounds == 0


class TestPdfOptions:
    def test_conversation_appendix_can_be_disabled(self):
        filled = Orchestrator(generate_pdf=False)
        assert filled.pdf_include_conversation is True

        pymupdf = pytest.importorskip("fitz")
        orchestrator = Orchestrator(pdf_include_conversation=False, max_revision_rounds=0)
        result = orchestrator.run("MSFT", date(2024, 1, 1), date(2024, 5, 1))
        text = "\n".join(page.get_text() for page in pymupdf.open(result.pdf_path))
        assert "Appendix — Agent Conversation Log" not in text
        assert "Executive Summary" in text


class TestOfflineCrew:
    def test_heuristic_recommendation_shape(self, kpis, sentiment):
        verdict = heuristic_recommendation(kpis, sentiment.to_dict())
        assert verdict["recommendation"] in VALID_RECOMMENDATIONS
        assert 20 <= verdict["confidence"] <= 85
        assert verdict["rationale"]

    def test_heuristic_is_bullish_for_strong_uptrend(self):
        bullish = {
            "period_roi_pct": 60.0,
            "relative_return_pct": 25.0,
            "momentum_20d_pct": 18.0,
            "price_vs_sma50_pct": 14.0,
            "rsi_14": 62.0,
            "sharpe_ratio": 2.4,
            "drawdown_pct": 4.0,
            "pe_ratio": 12.0,
            "is_demo": False,
            "trading_days": 250,
        }
        verdict = heuristic_recommendation(bullish, {"score": 0.6, "label": "bullish"})
        assert verdict["recommendation"] in {"BUY", "STRONG BUY"}

    def test_heuristic_is_bearish_for_weak_downtrend(self):
        bearish = {
            "period_roi_pct": -35.0,
            "relative_return_pct": -22.0,
            "momentum_20d_pct": -15.0,
            "price_vs_sma50_pct": -12.0,
            "rsi_14": 28.0,
            "sharpe_ratio": -1.6,
            "max_drawdown_pct": 41.0,
            "volatility_annual_pct": 62.0,
            "pe_ratio": 60.0,
            "is_demo": False,
            "trading_days": 250,
        }
        verdict = heuristic_recommendation(bearish, {"score": -0.55, "label": "bearish"})
        assert verdict["recommendation"] in {"SELL", "STRONG SELL"}

    def test_briefs_are_markdown(self, kpis, sentiment, news):
        research = research_brief(news, sentiment.to_dict(), "AAPL", "Apple Inc.")
        assert research.startswith("## News Summary")
        assert "## Sentiment" in research

        analysis = analyst_brief(kpis, "AAPL", "Apple Inc.")
        assert "## KPI Highlights" in analysis
        assert analysis.count("|") > 20  # table rows

    def test_compose_report_has_all_sections(self, kpis, sentiment, news, date_range):
        start, end = date_range
        report = compose_report(
            ticker="AAPL",
            company_name="Apple Inc.",
            kpis=kpis,
            sentiment=sentiment.to_dict(),
            news=news,
            start=start,
            end=end,
        )
        for key in ("executive_summary", "kpi_table", "recommendation", "confidence", "disclaimer"):
            assert has_section(report, key)


class TestReportQa:
    def test_split_sections(self):
        sections = split_sections("# T\n\n## A\nbody a\n\n## B\nbody b")
        assert sections["a"] == "body a"
        assert sections["b"] == "body b"

    def test_ensure_report_sections_fills_gaps(self, kpis, sentiment, news, date_range):
        start, end = date_range
        fallback = compose_report(
            ticker="AAPL",
            company_name="Apple Inc.",
            kpis=kpis,
            sentiment=sentiment.to_dict(),
            news=news,
            start=start,
            end=end,
        )
        partial = "# AAPL\n\n## Executive Summary\nShort summary."
        merged = ensure_report_sections(partial, fallback, kpis)
        assert has_section(merged, "executive_summary")
        assert has_section(merged, "recommendation")
        assert has_section(merged, "disclaimer")
        assert "Appendix" in merged

    def test_ensure_report_sections_returns_fallback_for_empty_draft(self, kpis):
        assert ensure_report_sections("", "# Fallback", kpis) == "# Fallback"

    def test_review_flags_missing_sections(self, kpis, sentiment):
        feedback = review_draft("Too short.", kpis, sentiment, is_demo=True)
        assert feedback and "too short" in feedback.lower()

    def test_review_passes_a_complete_report(self, kpis, sentiment, news, date_range):
        start, end = date_range
        report = compose_report(
            ticker="AAPL",
            company_name="Apple Inc.",
            kpis=kpis,
            sentiment=sentiment.to_dict(),
            news=news,
            start=start,
            end=end,
        )
        assert review_draft(report, kpis, sentiment, is_demo=kpis["is_demo"]) is None

    def test_review_flags_missing_disclosure(self, kpis, sentiment):
        filler = (
            "This analysis window covers 1.2 months of trading with 3.4% average "
            "daily range and a 5.6% drawdown, 7.8 beta-adjusted return, 9.1 turnover.\n"
        )
        report = (
            "# AAPL\n## Executive Summary\nA summary with 1.2 and 3.4 and 5.6 numbers.\n"
            + filler * 3
            + "## KPI Table\n| a | b |\n| 1.1 | 2.2 |\n## Recommendation\n**BUY** on 12.5 momentum\n"
            "## Confidence Score\n**Confidence:** 66/100\n## Disclaimer\nNot advice.\n"
        )
        feedback = review_draft(report, kpis, sentiment, is_demo=True)
        assert feedback is not None
        assert "synthetic" in feedback.lower()
