"""Contract tests for the agent prompts and the report structure.

The prompts are the product's specification: if someone edits an instruction and
silently drops a required section, these tests fail. They also verify that the
canonical headings in the prompts, the aliases used by the QA/PDF layers, and the
deterministic report all agree.
"""

from __future__ import annotations

import pytest

from agents.offline_crew import compose_report, research_brief
from agents.orchestrator import has_section, review_draft, split_sections
from prompts.prompts import (
    ANALYST,
    COMMON_RULES,
    DISCLAIMER_TEXT,
    EXPECTED_OUTPUT,
    ORCHESTRATOR,
    REPORT_SECTIONS,
    REQUIRED_SECTIONS,
    RESEARCHER,
    REVISION_REVIEW_PROMPT,
    SECTION_ALIASES,
    WRITER,
    analyst_task,
    orchestrator_instructions,
    researcher_task,
    revision_task,
    team_task,
    writer_task,
)

CANONICAL_HEADINGS = [
    "Executive Summary",
    "Market News Summary",
    "KPI Table and Charts",
    "Sentiment Overview",
    "Analysis and Outlook",
    "Recommendation",
    "Disclaimer",
]


class TestReportContract:
    def test_sections_match_the_spec(self):
        assert [heading for heading, _ in REPORT_SECTIONS] == CANONICAL_HEADINGS

    def test_every_section_has_aliases(self):
        for _, key in REPORT_SECTIONS:
            assert SECTION_ALIASES.get(key), f"no aliases for {key}"
        # The canonical heading itself must always be an alias.
        for heading, key in REPORT_SECTIONS:
            assert heading.lower() in SECTION_ALIASES[key]

    def test_required_sections_are_a_subset(self):
        keys = {key for _, key in REPORT_SECTIONS}
        for _, key in REQUIRED_SECTIONS:
            assert key in keys

    def test_writer_prompt_contains_every_heading(self):
        for heading in CANONICAL_HEADINGS:
            assert f"## {heading}" in WRITER, f"Writer prompt is missing '{heading}'"

    def test_writer_prompt_requires_label_and_confidence_line(self):
        assert "**BUY**" in WRITER and "**HOLD**" in WRITER and "**SELL**" in WRITER
        assert "**Confidence:** NN/100" in WRITER

    def test_writer_prompt_forbids_guessing(self):
        assert "request clarification" in WRITER.lower()
        assert "not financial advice" in WRITER.lower()

    def test_researcher_prompt_requirements(self):
        for phrase in ("Positive", "Neutral", "Negative", "5-10 key news items", "-1 to +1"):
            assert phrase in RESEARCHER
        assert "Never invent news" in RESEARCHER
        assert "say so explicitly" in RESEARCHER
        # Tool names the agent must call.
        assert "search_stock_news" in RESEARCHER
        assert "analyze_news_sentiment" in RESEARCHER

    @pytest.mark.parametrize(
        "metric", ["P/E", "EPS", "market cap", "ROI", "volatility", "50/200-day", "RSI", "drawdown"]
    )
    def test_analyst_prompt_lists_requested_metrics(self, metric):
        assert metric.lower() in ANALYST.lower()

    def test_analyst_prompt_requires_table_and_chart_paths(self):
        assert "markdown table of KPIs" in ANALYST
        assert "chart file paths" in ANALYST
        assert "never guess numbers" in ANALYST.lower()

    def test_orchestrator_prompt_order_and_loop_limit(self):
        instructions = orchestrator_instructions(max_revisions=2)
        assert "Market Researcher gathers news" in instructions
        assert "Data Analyst computes KPIs" in instructions
        assert "Report Writer produces" in instructions
        assert "max 2 revision rounds" in instructions
        assert "Log every hand-off" in instructions

    def test_orchestrator_limit_follows_configuration(self):
        assert "max 3 revision rounds" in orchestrator_instructions(max_revisions=3)

    def test_common_rules_are_attached_to_every_agent(self):
        for prompt in (RESEARCHER, ANALYST, WRITER, ORCHESTRATOR):
            assert COMMON_RULES.strip() in prompt

    def test_review_checklist_matches_headings(self):
        for heading in CANONICAL_HEADINGS:
            assert heading in REVISION_REVIEW_PROMPT

    def test_expected_output_lists_headings(self):
        for heading in CANONICAL_HEADINGS:
            assert heading in EXPECTED_OUTPUT

    def test_disclaimer_starts_with_the_required_sentence(self):
        assert DISCLAIMER_TEXT.startswith("This is not financial advice.")


class TestTaskTemplates:
    def test_researcher_task(self):
        task = researcher_task("AAPL", "Apple Inc.", "2024-01-01", "2024-06-30")
        assert "AAPL" in task and "Apple Inc." in task
        assert "2024-01-01" in task
        assert "Positive / Neutral / Negative" in task

    def test_analyst_task(self):
        task = analyst_task("AAPL", "Apple Inc.", "2024-01-01", "2024-06-30")
        assert "P/E" in task and "RSI" in task and "drawdown" in task

    def test_writer_task_embeds_upstream_output(self):
        task = writer_task(
            "AAPL",
            "Apple Inc.",
            "2024-01-01",
            "2024-06-30",
            research="RESEARCH-MARKER",
            analysis="ANALYSIS-MARKER",
        )
        assert "RESEARCH-MARKER" in task
        assert "ANALYSIS-MARKER" in task
        assert "Market News Summary" in task

    def test_writer_task_handles_missing_upstream_output(self):
        task = writer_task("AAPL", "", "2024-01-01", "2024-06-30", "", "")
        assert "no research output" in task.lower()

    def test_team_task(self):
        task = team_task("TSLA", "Tesla", "2024-01-01", "2024-03-01")
        assert "TSLA" in task and "revision loop" in task

    def test_revision_task(self):
        task = revision_task(feedback="add a confidence score", draft="DRAFT-BODY")
        assert "add a confidence score" in task
        assert "DRAFT-BODY" in task


class TestDeterministicReportMatchesTheContract:
    def test_research_brief_matches_researcher_prompt(self, news, sentiment):
        brief = research_brief(news, sentiment.to_dict(), "AAPL", "Apple Inc.")
        for heading in ("## Key News Items", "## Sentiment", "## Key Risks", "## Catalysts"):
            assert heading in brief
        # 5-10 items and one classification per item.
        item_lines = [
            line
            for line in brief.split("## Key News Items")[1].split("## Sentiment")[0].splitlines()
            if line.startswith("- ")
        ]
        assert 1 <= len(item_lines) <= 10
        assert all(
            ("**Positive**" in line or "**Neutral**" in line or "**Negative**" in line)
            for line in item_lines
        )

    def test_report_sections_and_order(self, kpis, sentiment, news, date_range):
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
        headings = [
            line.lstrip("# ").strip() for line in report.splitlines() if line.startswith("## ")
        ]
        assert headings == CANONICAL_HEADINGS
        assert "**Confidence:**" in split_sections(report)["recommendation"]
        assert "This is not financial advice." in report
        assert "**HOLD**" in report or "**BUY**" in report or "**SELL**" in report

    def test_report_passes_its_own_qa(self, kpis, sentiment, news, date_range):
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

    def test_kpi_table_has_at_least_ten_rows(self, kpis, sentiment, news, date_range):
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
        table_rows = [
            line
            for line in split_sections(report)["kpi table and charts"].splitlines()
            if line.startswith("|")
        ]
        assert len(table_rows) >= 12  # header + separator + 10+ metrics

    def test_all_required_sections_are_detectable(self, kpis, sentiment, news, date_range):
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
        for _, key in REQUIRED_SECTIONS:
            assert has_section(report, key)
