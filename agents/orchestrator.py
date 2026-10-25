"""Orchestration: run the three-agent crew and assemble every deliverable.

Two execution paths share one contract (:class:`AnalysisResult`):

1. **Live crew (primary).** An agno ``Team`` in ``coordinate`` mode drives
   Market Researcher → Data Analyst → Report Writer. The orchestrator then
   reviews the draft with :func:`agents.offline_crew.heuristic_recommendation`
   style checks and *optionally* loops once back to the writer with concrete
   revision instructions.
2. **Offline crew (fallback).** Without ``GOOGLE_API_KEY`` (or if the API call
   fails) the same data is turned into the same three artefacts deterministically
   by :mod:`agents.offline_crew`, so the UI, PDF and tests keep working.

Both paths start by fetching market data, news, sentiment and charts directly.
That is deliberate: the PDF must always contain real (or clearly labelled demo)
tables and images, even if the LLM ignores its tools or times out.
"""

from __future__ import annotations

import re
import time
import uuid
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import date, datetime
from typing import Any

from agents.base import response_text
from agents.llm import LLMConfigurationError, build_model
from agents.offline_crew import (
    analyst_brief,
    compose_report,
    heuristic_recommendation,
    research_brief,
)
from config.settings import get_settings
from prompts.prompts import (
    EXPECTED_OUTPUT,
    REQUIRED_SECTIONS,
    SECTION_ALIASES,
    orchestrator_instructions,
    revision_task,
    team_task,
)
from tools.chart_tools import make_charts, save_chart_images
from tools.finance_tools import (
    DEFAULT_BENCHMARK,
    StockData,
    compute_kpis,
    fetch_benchmark,
    fetch_stock_data,
)
from tools.news_tools import search_news
from tools.sentiment import analyze_texts
from utils.formatting import safe_float
from utils.logger import ConversationLog, get_logger
from utils.parsing import (
    normalize_ticker,
    parse_recommendation,
    validate_date_range,
)

logger = get_logger(__name__)

ProgressCallback = Callable[[str, float, str], None]


@dataclass
class AnalysisResult:
    """Everything one analysis run produces."""

    ticker: str
    company_name: str
    start: date
    end: date
    report_markdown: str
    kpis: dict[str, Any] = field(default_factory=dict)
    sentiment: dict[str, Any] = field(default_factory=dict)
    news: list[dict[str, Any]] = field(default_factory=list)
    chart_images: dict[str, str] = field(default_factory=dict)
    figures: dict[str, Any] = field(default_factory=dict)
    recommendation: str | None = None
    confidence: int | None = None
    confidence_rationale: str | None = None
    revision_rounds: int = 0
    is_demo: bool = False
    llm_used: bool = False
    warnings: list[str] = field(default_factory=list)
    conversation: list[dict[str, Any]] = field(default_factory=list)
    pdf_path: str | None = None
    log_path: str | None = None
    report_path: str | None = None
    run_id: str = ""
    duration_s: float = 0.0
    timings: dict[str, float] = field(default_factory=dict)
    mode: str = "live"

    @property
    def summary_rows(self) -> list[dict[str, str]]:
        """Top-level facts for a quick Streamlit metrics row."""
        from utils.formatting import fmt_confidence, fmt_currency, fmt_pct

        return [
            {"label": "Last price", "value": fmt_currency(self.kpis.get("current_price"))},
            {"label": "Period ROI", "value": fmt_pct(self.kpis.get("period_roi_pct"))},
            {"label": "Sentiment", "value": str(self.sentiment.get("label", "n/a")).title()},
            {"label": "Recommendation", "value": self.recommendation or "n/a"},
            {"label": "Confidence", "value": fmt_confidence(self.confidence)},
        ]


class Orchestrator:
    """Coordinates the agent crew for a single analysis request."""

    def __init__(
        self,
        *,
        model: Any | None = None,
        log: ConversationLog | None = None,
        progress: ProgressCallback | None = None,
        allow_llm: bool = True,
        generate_pdf: bool = True,
        max_revision_rounds: int | None = None,
        force_demo: bool | None = None,
        pdf_include_conversation: bool = True,
    ) -> None:
        self.settings = get_settings()
        self.max_revisions = (
            self.settings.max_revision_rounds
            if max_revision_rounds is None
            else max(0, int(max_revision_rounds))
        )
        self.force_demo = force_demo
        self.pdf_include_conversation = pdf_include_conversation
        self.log = log or ConversationLog()
        self.progress = progress
        self.allow_llm = allow_llm
        self.should_generate_pdf = generate_pdf
        self.model = model
        self.llm_error: str | None = None

    # ------------------------------------------------------------------ utils
    def _report(self, stage: str, fraction: float, message: str) -> None:
        logger.info("[%s] %s", stage, message)
        if self.progress is not None:
            try:
                self.progress(stage, max(0.0, min(1.0, fraction)), message)
            except Exception:  # noqa: BLE001 - a UI callback must never break a run
                logger.debug("Progress callback failed", exc_info=True)

    def _model(self) -> Any | None:
        """Return a Gemini model, or None when the crew must run offline."""
        if not self.allow_llm:
            self.llm_error = "LLM disabled for this run"
            return None
        if self.model is not None:
            return self.model
        try:
            self.model = build_model()
            return self.model
        except LLMConfigurationError as exc:
            self.llm_error = str(exc)
            logger.warning("Running without an LLM: %s", exc)
            return None

    # -------------------------------------------------------------------- run
    def run(
        self,
        ticker: str,
        start: date | str,
        end: date | str,
        *,
        use_llm: bool | None = None,
    ) -> AnalysisResult:
        """Run the full analysis.

        Args:
            ticker: Stock symbol, e.g. ``"AAPL"``.
            start: Analysis window start.
            end: Analysis window end.
            use_llm: Force the live crew on/off. ``None`` decides automatically.

        Returns:
            A populated :class:`AnalysisResult` (including a PDF path when possible).

        Raises:
            InputValidationError: For an invalid ticker or date range.
            FinanceDataError: When market data is unavailable and demo fallback is off.
        """
        run_started = time.perf_counter()
        run_id = uuid.uuid4().hex[:8]
        timings: dict[str, float] = {}
        warnings: list[str] = []

        symbol = normalize_ticker(ticker)
        start_date, end_date = validate_date_range(start, end)
        self._report("Input", 0.02, f"Analysing {symbol} from {start_date} to {end_date}")
        self.log.add(
            "Orchestrator",
            "started",
            f"Analysis requested for {symbol} ({start_date} → {end_date})",
            run_id=run_id,
        )

        if use_llm is not None:
            self.allow_llm = use_llm
        model = self._model()
        if model is None:
            warnings.append(
                f"Live LLM crew unavailable ({self.llm_error or 'no model'}). "
                "Used the deterministic offline crew instead."
            )

        # ---------------------------------------------------------- data phase
        phase_started = time.perf_counter()
        self._report("Data", 0.08, "Downloading price history and company profile…")
        stock = fetch_stock_data(symbol, start_date, end_date, force_demo=self.force_demo)
        benchmark = None
        if not stock.is_demo:
            benchmark = fetch_benchmark(DEFAULT_BENCHMARK, start_date, end_date)
        kpis = compute_kpis(stock, benchmark)
        timings["market_data"] = round(time.perf_counter() - phase_started, 2)
        warnings.extend(stock.warnings)
        warnings.extend(kpis.get("data_quality", [])[:2])
        self.log.tool_call(
            "Data Analyst",
            "get_financial_kpis",
            f"{kpis['trading_days']} trading days, ROI {kpis.get('period_roi_pct')}%, "
            f"RSI {kpis.get('rsi_14')}",
            demo=stock.is_demo,
        )

        self._report("Data", 0.25, "Collecting recent news…")
        phase_started = time.perf_counter()
        news = self._fetch_news(symbol, stock)
        sentiment = analyze_texts(news)
        timings["news"] = round(time.perf_counter() - phase_started, 2)
        self.log.tool_call(
            "Market Researcher",
            "search_stock_news",
            f"{len(news)} headlines, sentiment={sentiment.label} ({sentiment.score:+.2f})",
        )

        self._report("Charts", 0.4, "Building charts…")
        phase_started = time.perf_counter()
        figures, chart_warnings = make_charts(stock.history, kpis, symbol, stock.currency)
        images, image_warnings = save_chart_images(
            figures, stock.history, kpis, symbol, stock.currency, run_id=run_id
        )
        warnings.extend(chart_warnings)
        warnings.extend(image_warnings)
        timings["charts"] = round(time.perf_counter() - phase_started, 2)
        self.log.add(
            "Data Analyst",
            "tool",
            f"make_charts — {len(figures)} figures, {len(images)} images saved",
        )

        # ------------------------------------------------------ drafting phase
        research_md = ""
        analysis_md = ""
        draft = ""
        revision_rounds = 0
        llm_used = False
        mode = "offline"

        if model is not None:
            self._report("Agents", 0.5, "Market Researcher → Data Analyst → Report Writer…")
            phase_started = time.perf_counter()
            try:
                research_md, analysis_md, draft, revision_rounds = self._run_live_crew(
                    model=model,
                    ticker=symbol,
                    company=stock.company_name,
                    start=start_date,
                    end=end_date,
                    stock=stock,
                    kpis=kpis,
                    news=news,
                    sentiment=sentiment,
                )
                rejection = validate_llm_draft(draft)
                if rejection:
                    self.log.error("Orchestrator", f"Rejected the live draft: {rejection}")
                    warnings.append(
                        f"The live crew returned an unusable response ({rejection}). "
                        "Used the deterministic offline crew instead."
                    )
                    draft = ""
                else:
                    llm_used = True
                    mode = "live"
            except Exception as exc:  # noqa: BLE001 - degrade, never crash the UI
                logger.exception("Live crew failed; using the offline crew")
                warnings.append(
                    f"Live agent crew failed ({type(exc).__name__}: {exc}). "
                    "Fell back to the deterministic offline crew."
                )
                self.log.error("Orchestrator", f"Live crew failed: {exc}")
            timings["agents"] = round(time.perf_counter() - phase_started, 2)

        if not draft.strip():
            self._report("Agents", 0.62, "Composing the report with the offline crew…")
            phase_started = time.perf_counter()
            research_md = research_md or research_brief(
                news, sentiment.to_dict(), symbol, stock.company_name
            )
            analysis_md = analysis_md or analyst_brief(kpis, symbol, stock.company_name)
            self.log.agent_done("Market Researcher", "Research brief prepared (offline crew)", 0.0)
            self.log.agent_done("Data Analyst", "Analyst brief prepared (offline crew)", 0.0)
            draft = compose_report(
                ticker=symbol,
                company_name=stock.company_name,
                kpis=kpis,
                sentiment=sentiment.to_dict(),
                news=news,
                start=start_date,
                end=end_date,
            )
            self.log.agent_done("Report Writer", "Draft report composed (offline crew)", 0.0)
            timings["offline_compose"] = round(time.perf_counter() - phase_started, 2)
            mode = "offline"

        # ----------------------------------------------------------- assembly
        self._report("Review", 0.75, "Reviewing the draft and enforcing report structure…")
        fallback_report = compose_report(
            ticker=symbol,
            company_name=stock.company_name,
            kpis=kpis,
            sentiment=sentiment.to_dict(),
            news=news,
            start=start_date,
            end=end_date,
        )
        report_md = ensure_report_sections(draft, fallback_report, kpis)

        verdict = parse_recommendation(report_md)
        heuristic = heuristic_recommendation(kpis, sentiment.to_dict())
        recommendation = verdict.get("recommendation") or heuristic["recommendation"]
        confidence = verdict.get("confidence")
        if confidence is None:
            confidence = heuristic["confidence"]
            self.log.warning(
                "Report Writer",
                "No numeric confidence score found in the report; used the heuristic score.",
            )
        if stock.is_demo:
            confidence = min(confidence, 65)
        rationale = verdict.get("rationale") or heuristic["rationale"]

        # -------------------------------------------------------------- files
        result = AnalysisResult(
            ticker=symbol,
            company_name=stock.company_name,
            start=start_date,
            end=end_date,
            report_markdown=report_md,
            kpis=kpis,
            sentiment=sentiment.to_dict(),
            news=news,
            chart_images=images,
            figures=figures,
            recommendation=recommendation,
            confidence=confidence,
            confidence_rationale=rationale,
            revision_rounds=revision_rounds,
            is_demo=stock.is_demo,
            llm_used=llm_used,
            warnings=_dedupe(warnings),
            conversation=self.log.to_records(),
            run_id=run_id,
            timings=timings,
            mode=mode,
        )

        self._report("Files", 0.85, "Saving report files…")
        result.report_path, result.log_path = self._save_text_artifacts(result)

        if self.should_generate_pdf:
            self._report("PDF", 0.92, "Generating the PDF report…")
            phase_started = time.perf_counter()
            try:
                from reports.pdf_generator import generate_pdf_report

                result.pdf_path = str(
                    generate_pdf_report(
                        result=result,
                        conversation=self.log if self.pdf_include_conversation else None,
                        include_conversation=self.pdf_include_conversation,
                    )
                )
                self.log.add("Report Writer", "PDF", f"PDF saved to {result.pdf_path}")
            except Exception as exc:  # noqa: BLE001 - PDF failure must not lose the analysis
                logger.exception("PDF generation failed")
                result.warnings.append(f"PDF generation failed: {type(exc).__name__}: {exc}")
                result.warnings = _dedupe(result.warnings)
            timings["pdf"] = round(time.perf_counter() - phase_started, 2)

        result.conversation = self.log.to_records()
        result.duration_s = round(time.perf_counter() - run_started, 2)
        self.log.agent_done(
            "Orchestrator",
            f"Analysis complete ({result.recommendation}, confidence {result.confidence})",
            result.duration_s,
        )
        result.conversation = self.log.to_records()
        # Persist the final conversation (it now includes the PDF + finish events).
        if result.log_path:
            try:
                self.log.save(result.log_path)
            except OSError:
                logger.debug("Could not update conversation log file", exc_info=True)

        self._report("Done", 1.0, f"Report ready in {result.duration_s:.1f}s")
        return result

    # ------------------------------------------------------------- internals
    def _fetch_news(self, ticker: str, stock: StockData) -> list[dict[str, Any]]:
        """Fetch news, degrading to an empty list instead of failing the run."""
        if stock.is_demo:
            # Never mix synthetic prices with real headlines.
            from tools.demo_data import demo_news

            items = demo_news(ticker, stock.company_name, self.settings.max_news_results)
            self.log.warning("Market Researcher", "News is synthetic to match demo market data.")
            return items
        try:
            return search_news(ticker, stock.company_name)
        except Exception as exc:  # noqa: BLE001
            logger.warning("News fetch failed for %s: %s", ticker, exc)
            self.log.warning("Market Researcher", f"News unavailable: {exc}")
            return []

    def _run_live_crew(
        self,
        *,
        model: Any,
        ticker: str,
        company: str,
        start: date,
        end: date,
        stock: StockData,
        kpis: dict[str, Any],
        news: list[dict[str, Any]],
        sentiment: Any,
    ) -> tuple[str, str, str, int]:
        """Execute the agno Team, then run the review/revision loop.

        Returns:
            ``(research_markdown, analysis_markdown, final_report, revision_rounds)``.
        """
        from agno.team import Team

        from agents.analyst import build_analyst
        from agents.researcher import build_researcher
        from agents.writer import build_writer

        researcher = build_researcher(model)
        analyst = build_analyst(model)
        writer = build_writer(model)

        self.log.agent_start(
            "Orchestrator",
            f"Assembling the crew: {researcher.name}, {analyst.name}, {writer.name}",
        )

        context = (
            f"\n\nPre-computed ground truth for this run (use these exact figures):\n"
            f"- Company: {company} | Sector: {stock.sector} | Currency: {stock.currency}\n"
            f"- Window: {start} → {end} ({kpis.get('trading_days')} trading days)\n"
            f"- Latest close: {kpis.get('current_price')} | Period ROI: {kpis.get('period_roi_pct')}%\n"
            f"- Volatility: {kpis.get('volatility_annual_pct')}% | RSI(14): {kpis.get('rsi_14')}\n"
            f"- SMA20/50: {kpis.get('sma_20')}/{kpis.get('sma_50')} | P/E: {kpis.get('pe_ratio')}\n"
            f"- Sentiment: {sentiment.label} ({sentiment.score:+.2f}) across "
            f"{sentiment.articles_analyzed} headlines\n"
            f"- Synthetic demo data: {stock.is_demo}\n"
            "- Charts already rendered by the system: price, volume, RSI, KPI bar chart\n"
        )

        team = Team(
            name="Financial Analysis Crew",
            mode=self.settings.team_mode,
            model=model,
            members=[researcher, analyst, writer],
            instructions=[orchestrator_instructions(self.max_revisions)],
            expected_output=EXPECTED_OUTPUT,
            show_members_responses=self.settings.show_members_responses,
            markdown=True,
            add_datetime_to_context=True,
            max_iterations=self.settings.team_max_iterations,
            debug_mode=False,
        )

        task = team_task(ticker, company, start, end) + context
        started = time.perf_counter()
        self.log.agent_start(
            "Orchestrator", "Delegating the analysis to the crew (coordinate mode)"
        )
        team_output = team.run(task)
        draft = response_text(team_output)
        self.log.agent_done(
            "Financial Analysis Crew",
            f"Team produced {len(draft)} characters of report content",
            time.perf_counter() - started,
        )

        # Member transcripts are recorded by agno inside the team output; we log a
        # compact digest so the conversation log stays readable in the UI.
        self._log_member_digest(team_output)

        rejection = validate_llm_draft(draft)
        if rejection:
            self.log.error("Financial Analysis Crew", f"Draft rejected: {rejection}")
            raise RuntimeError(f"The live crew returned an unusable response — {rejection}")

        # -- revision loop ----------------------------------------------------
        revision_rounds = 0
        current = draft
        for round_index in range(1, self.max_revisions + 1):
            feedback = review_draft(current, kpis, sentiment, stock.is_demo)
            if not feedback:
                self.log.add(
                    "Orchestrator", "review", f"Revision round {round_index}: draft approved"
                )
                break

            revision_rounds = round_index
            self.log.warning("Orchestrator", f"Revision round {round_index} requested: {feedback}")
            self._report(
                "Revision",
                0.68,
                f"Revision round {round_index}: sending the draft back to the Report Writer",
            )
            try:
                revised = writer.run(
                    revision_task(feedback=feedback, draft=current)
                    + "\n\nSupporting research:\n"
                    + research_digest(news, sentiment)
                    + "\n\nSupporting analytics:\n"
                    + analyst_brief(kpis, ticker, company)
                )
                revised_text = response_text(revised)
            except Exception as exc:  # noqa: BLE001
                self.log.error("Report Writer", f"Revision failed: {exc}")
                break

            rejection = validate_llm_draft(revised_text)
            if rejection:
                self.log.warning(
                    "Report Writer",
                    f"Revision rejected ({rejection}); keeping the previous draft",
                )
                break
            current = revised_text
            self.log.agent_done("Report Writer", f"Revision {round_index} delivered", 0.0)

        final_report = ensure_report_sections(
            current,
            compose_report(
                ticker=ticker,
                company_name=company,
                kpis=kpis,
                sentiment=sentiment.to_dict(),
                news=news,
                start=start,
                end=end,
            ),
            kpis,
        )

        return (
            research_digest(news, sentiment),
            analyst_brief(kpis, ticker, company),
            final_report,
            revision_rounds,
        )

    def _log_member_digest(self, team_output: Any) -> None:
        """Record a short line per member response found in the team output."""
        for message in getattr(team_output, "messages", []) or []:
            author = getattr(message, "agent_name", None) or getattr(message, "name", None)
            role = getattr(message, "role", "")
            content = response_text(message)
            if not author or not content or role in {"user", "system"}:
                continue
            preview = content.replace("\n", " ")[:160]
            self.log.add(str(author), "responded", f"{preview}…")

    def _save_text_artifacts(self, result: AnalysisResult) -> tuple[str, str]:
        """Write the Markdown report and conversation JSON to the output folder."""
        stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        base = self.settings.reports_dir
        base.mkdir(parents=True, exist_ok=True)
        report_path = base / f"{result.ticker}_{stamp}_{result.run_id}.md"
        log_path = base / f"{result.ticker}_{stamp}_{result.run_id}_conversation.json"
        try:
            report_path.write_text(result.report_markdown, encoding="utf-8")
            self.log.save(log_path)
        except OSError as exc:
            logger.warning("Could not write report artefacts: %s", exc)
            return "", ""
        return str(report_path), str(log_path)


# ---------------------------------------------------------------------------
# Report QA helpers
# ---------------------------------------------------------------------------
_HEADING_RE = re.compile(r"^(#{1,4})\s+(.*?)\s*$", re.MULTILINE)


def split_sections(markdown: str) -> dict[str, str]:
    """Split a Markdown document into ``{lowercase heading: body}``."""
    sections: dict[str, str] = {}
    matches = list(_HEADING_RE.finditer(markdown or ""))
    for index, match in enumerate(matches):
        heading = match.group(2).strip().lower()
        body_start = match.end()
        body_end = matches[index + 1].start() if index + 1 < len(matches) else len(markdown)
        sections[heading] = markdown[body_start:body_end].strip()
    return sections


def has_section(markdown: str, key: str) -> bool:
    """True when any alias for ``key`` appears as a heading in ``markdown``."""
    sections = split_sections(markdown)
    return any(alias in sections for alias in SECTION_ALIASES.get(key, (key,)))


def ensure_report_sections(draft: str, fallback: str, kpis: dict[str, Any]) -> str:
    """Guarantee the report contains every required section.

    Sections missing from the LLM draft are copied from the deterministic
    fallback report, so the PDF always has a summary, KPI table, recommendation,
    confidence score and disclaimer.
    """
    if not draft or not draft.strip():
        return fallback

    required = {
        "Executive Summary": "executive_summary",
        "KPI Table": "kpi_table",
        "Recommendation": "recommendation",
        "Confidence Score": "confidence",
        "Disclaimer": "disclaimer",
    }
    missing = [
        heading for heading in REQUIRED_SECTIONS if not has_section(draft, required[heading])
    ]

    if not draft.lstrip().startswith("# "):
        draft = f"# {kpis.get('ticker', '')} — Financial Analysis Report\n\n{draft.lstrip()}"
    if kpis.get("is_demo") and "SYNTHETIC DEMO DATA" not in draft.upper():
        draft = (
            draft.split("\n", 1)[0]
            + "\n\n> ⚠️ **SYNTHETIC DEMO DATA** — figures in this report are simulated and "
            "must not be used for investment decisions.\n"
            + (draft.split("\n", 1)[1] if "\n" in draft else "")
        )

    if not missing:
        return draft

    fallback_sections = split_sections(fallback)
    appendix = ["\n\n---\n", "## Appendix — system-generated sections\n"]
    for heading in missing:
        key = required[heading]
        body = None
        for alias in SECTION_ALIASES.get(key, ()):
            if alias in fallback_sections:
                body = fallback_sections[alias]
                break
        appendix.append(f"### {heading}\n{body or '_Section could not be generated._'}\n")
        logger.info("Appended missing report section: %s", heading)

    return draft.rstrip() + "\n" + "\n".join(appendix)


#: Substrings that indicate the provider/library returned an error *as content*
#: (agno surfaces some provider failures as the run's content rather than raising).
_LLM_FAILURE_MARKERS = (
    "error in team run",
    "error in agent run",
    "model provider error",
    "unknown error from gemini api",
    "api key not valid",
    "invalid api key",
    "permission_denied",
    "resource_exhausted",
    "unauthenticated",
    "service unavailable",
)


def validate_llm_draft(draft: str) -> str | None:
    """Check that an LLM draft is a genuine report, not an error page.

    agno can return provider failures as the run content instead of raising, so
    the orchestrator must inspect the text before trusting it.

    Args:
        draft: Markdown returned by the team/writer.

    Returns:
        ``None`` when the draft is usable, otherwise a short rejection reason.
    """
    text = (draft or "").strip()
    lowered = text.lower()
    for marker in _LLM_FAILURE_MARKERS:
        if marker in lowered:
            return f"draft contains a provider error ('{marker}')"
    if len(text) < 300:
        return f"draft is too short ({len(text)} characters)"
    core = {
        "executive_summary": has_section(text, "executive_summary"),
        "recommendation": has_section(text, "recommendation"),
        "kpi_table": has_section(text, "kpi_table"),
        "confidence": has_section(text, "confidence"),
    }
    if not (core["executive_summary"] and core["recommendation"]):
        return "draft is missing the executive summary or recommendation section"
    if sum(1 for present in core.values() if present) < 3:
        return "draft is missing too many required sections"
    return None


def review_draft(
    draft: str,
    kpis: dict[str, Any],
    sentiment: Any,
    is_demo: bool,
) -> str | None:
    """Deterministic QA of a draft report.

    Args:
        draft: Markdown report to review.
        kpis: Computed KPIs (used to detect contradictions).
        sentiment: A ``SentimentResult``-like object.
        is_demo: Whether the run used synthetic data.

    Returns:
        Specific revision instructions, or ``None`` when the draft passes.
    """
    if not draft or len(draft.strip()) < 400:
        return (
            "The report is too short. Expand every section, include the KPI table with at "
            "least 10 rows, and add the recommendation and confidence score."
        )

    problems: list[str] = []
    for heading, key in (
        ("Executive Summary", "executive_summary"),
        ("KPI Table", "kpi_table"),
        ("Recommendation", "recommendation"),
        ("Confidence Score", "confidence"),
        ("Disclaimer", "disclaimer"),
    ):
        if not has_section(draft, key):
            problems.append(f"add the missing '{heading}' section")

    parsed = parse_recommendation(draft)
    if not parsed.get("recommendation"):
        problems.append(
            "state a bold recommendation label (STRONG BUY / BUY / HOLD / SELL / STRONG SELL)"
        )
    if parsed.get("confidence") is None:
        problems.append("add a numeric confidence score in the form '**Confidence:** NN/100'")

    numbers = len(re.findall(r"\d+\.\d+|\d{1,3}%", draft))
    if numbers < 8:
        problems.append("cite more concrete figures from the KPI analysis (at least 8 numbers)")

    recommendation = parsed.get("recommendation")
    sentiment_score = safe_float(getattr(sentiment, "score", 0.0)) or 0.0
    if recommendation in {"STRONG BUY", "BUY"} and sentiment_score <= -0.35:
        problems.append(
            "explain why a BUY recommendation is justified despite clearly negative news sentiment"
        )
    if recommendation in {"STRONG SELL", "SELL"} and sentiment_score >= 0.35:
        problems.append(
            "explain why a SELL recommendation is justified despite clearly positive news sentiment"
        )
    if is_demo and "SYNTHETIC" not in draft.upper():
        problems.append("disclose that the figures are synthetic demo data")
    if is_demo and (parsed.get("confidence") or 0) > 85:
        problems.append("lower the confidence score: synthetic demo data cannot support >85")

    if not problems:
        return None
    return "Please revise the draft: " + "; ".join(problems) + "."


def research_digest(news: list[dict[str, Any]], sentiment: Any) -> str:
    """Compact research summary passed to the writer during revisions."""
    headlines = (
        "\n".join(
            f"- {item.get('title', '')} ({item.get('source', '')}, {item.get('published', '')})"
            for item in news[:6]
        )
        or "- no headlines available"
    )
    return (
        f"Sentiment: {sentiment.label} (score {sentiment.score:+.2f}; "
        f"{sentiment.positive_count}+ / {sentiment.negative_count}- / "
        f"{sentiment.neutral_count}= across {sentiment.articles_analyzed} items)\n"
        f"Headlines:\n{headlines}"
    )


def _dedupe(items: list[str]) -> list[str]:
    seen: set[str] = set()
    output: list[str] = []
    for item in items:
        text = str(item).strip()
        if text and text not in seen:
            seen.add(text)
            output.append(text)
    return output


# ---------------------------------------------------------------------------
# Convenience entry points
# ---------------------------------------------------------------------------
def run_analysis(
    ticker: str,
    start: date | str,
    end: date | str,
    *,
    progress: ProgressCallback | None = None,
    use_llm: bool | None = None,
    log: ConversationLog | None = None,
    generate_pdf: bool = True,
    max_revision_rounds: int | None = None,
    force_demo: bool | None = None,
    pdf_include_conversation: bool = True,
) -> AnalysisResult:
    """Run one complete analysis and return every artefact.

    This is the function the Streamlit app calls.

    Example
    -------
    >>> from agents.orchestrator import run_analysis
    >>> result = run_analysis("AAPL", "2024-01-01", "2024-06-30")  # doctest: +SKIP
    """
    orchestrator = Orchestrator(
        progress=progress,
        log=log,
        generate_pdf=generate_pdf,
        max_revision_rounds=max_revision_rounds,
        force_demo=force_demo,
        pdf_include_conversation=pdf_include_conversation,
    )
    return orchestrator.run(ticker, start, end, use_llm=use_llm)


__all__ = [
    "AnalysisResult",
    "EXPECTED_OUTPUT",
    "Orchestrator",
    "ensure_report_sections",
    "has_section",
    "research_digest",
    "review_draft",
    "run_analysis",
    "split_sections",
    "validate_llm_draft",
]
