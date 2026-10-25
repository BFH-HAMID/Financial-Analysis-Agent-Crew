"""Tests for chart rendering and PDF generation."""

from __future__ import annotations

from pathlib import Path

import pytest

from tools.chart_tools import (
    ChartError,
    build_kpi_chart,
    build_price_chart,
    build_volume_chart,
    make_charts,
    save_chart_images,
)


class TestCharts:
    def test_all_four_figures_are_built(self, stock, kpis):
        figures, warnings = make_charts(stock.history, kpis, "AAPL", stock.currency)
        assert set(figures) == {"price", "volume", "rsi", "kpi"}
        assert warnings == []

    def test_price_chart_has_traces(self, stock):
        fig = build_price_chart(stock.history, "AAPL")
        assert len(fig.data) >= 1
        assert "AAPL" in fig.layout.title.text

    def test_volume_chart_requires_volume(self, stock):
        history = stock.history.drop(columns=["Volume"])
        with pytest.raises(ChartError):
            build_volume_chart(history, "AAPL")

    def test_kpi_chart_needs_values(self):
        with pytest.raises(ChartError):
            build_kpi_chart({}, "AAPL")

    def test_empty_history_is_handled(self, kpis):
        import pandas as pd

        figures, warnings = make_charts(pd.DataFrame(), kpis, "AAPL")
        assert figures == {}
        assert warnings

    def test_png_export_writes_files(self, stock, kpis, tmp_path):
        figures, _ = make_charts(stock.history, kpis, "AAPL", stock.currency)
        images, warnings = save_chart_images(
            figures,
            stock.history,
            kpis,
            "AAPL",
            stock.currency,
            output_dir=tmp_path,
            run_id="testrun",
        )
        assert set(images) == {"price", "volume", "rsi", "kpi"}
        for path in images.values():
            assert Path(path).exists()
            assert Path(path).stat().st_size > 1000
        assert not [w for w in warnings if "Could not render" in w]


class TestPdf:
    def test_pdf_is_generated(self, analysis_result):
        assert analysis_result.pdf_path, "expected a generated PDF path"
        pdf = Path(analysis_result.pdf_path)
        assert pdf.exists()
        assert pdf.stat().st_size > 20_000
        assert pdf.read_bytes()[:5] == b"%PDF-"

    def test_pdf_contains_expected_text(self, analysis_result):
        pymupdf = pytest.importorskip("fitz")
        document = pymupdf.open(analysis_result.pdf_path)
        text = "\n".join(page.get_text() for page in document)
        for expected in (
            "Financial Analysis Report",
            "Executive Summary",
            "KPI Table",
            "Recommendation",
            "Confidence Score",
            "Disclaimer",
        ):
            assert expected in text, f"{expected!r} missing from the PDF"
        assert "SYNTHETIC DEMO" in text.upper()

    def test_pdf_includes_charts_and_conversation(self, analysis_result):
        pymupdf = pytest.importorskip("fitz")
        document = pymupdf.open(analysis_result.pdf_path)
        assert document.page_count >= 3
        images = sum(len(page.get_images()) for page in document)
        assert images >= 3, "charts should be embedded"
        text = "\n".join(page.get_text() for page in document)
        assert "Agent Conversation Log" in text

    def test_pdf_without_conversation_appendix(self, analysis_result):
        from reports.pdf_generator import generate_pdf_report

        path = generate_pdf_report(analysis_result, conversation=None, include_conversation=False)
        assert Path(path).exists()


class TestPdfHelpers:
    def test_inline_markup_bold_and_code(self):
        from reports.pdf_generator import inline_markup

        assert inline_markup("**bold**") == "<b>bold</b>"
        assert "Courier" in inline_markup("`code`")

    def test_inline_markup_never_touches_identifiers(self):
        from reports.pdf_generator import inline_markup

        assert "ALLOW_DEMO_DATA" in inline_markup("ALLOW_DEMO_DATA=1")

    def test_inline_markup_escapes_unknown_tags(self):
        from reports.pdf_generator import inline_markup

        assert "&lt;script&gt;" in inline_markup("<script>x</script>")
        assert "<b>" in inline_markup("<b>kept</b>")

    def test_sanitize_transliterates_unicode(self):
        from reports.pdf_generator import sanitize_text

        assert sanitize_text("a → b – c") == "a -> b - c"
        assert "✅" not in sanitize_text("✅ ok")

    def test_markdown_to_flowables_handles_tables_and_lists(self):
        from reports.pdf_generator import _build_styles, markdown_to_flowables

        markdown = (
            "Intro paragraph.\n\n"
            "- first bullet\n- second bullet\n\n"
            "| Metric | Value |\n| --- | --- |\n| P/E | 21.4 |\n\n"
            "> a callout\n"
        )
        flowables = markdown_to_flowables(markdown, _build_styles())
        assert len(flowables) >= 4

    def test_section_split_and_alias_lookup(self):
        from reports.pdf_generator import find_section, split_sections

        sections = split_sections("## Executive Summary\nGood.\n\n## Risks\nHigh beta.")
        assert find_section(sections, "executive_summary") == "Good."
        assert find_section(sections, "risk_assessment") == "High beta."
        assert find_section(sections, "disclaimer") is None
