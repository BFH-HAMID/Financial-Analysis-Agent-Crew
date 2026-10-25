"""Financial Analysis Agent Crew — Streamlit front end.

Run with::

    streamlit run app.py

The UI is a thin layer over :func:`agents.orchestrator.run_analysis`. It owns no
business logic: it validates input, shows progress, renders results and offers
the PDF download.
"""

from __future__ import annotations

import os
from datetime import date, timedelta

import pandas as pd
import streamlit as st


# ---------------------------------------------------------------------------
# Streamlit Cloud secrets -> environment (must happen before config import)
# ---------------------------------------------------------------------------
def _load_streamlit_secrets() -> None:
    """Copy well-known keys from ``st.secrets`` into the environment once."""
    keys = (
        "GOOGLE_API_KEY",
        "NEWS_API_KEY",
        "GEMINI_MODEL",
        "SENTIMENT_BACKEND",
        "MAX_REVISION_ROUNDS",
        "ALLOW_DEMO_DATA",
        "FORCE_DEMO_DATA",
        "LOG_LEVEL",
    )
    try:
        secrets = st.secrets  # may raise when no secrets.toml exists
    except Exception:  # noqa: BLE001
        return
    for key in keys:
        try:
            if key not in os.environ and key in secrets:
                os.environ[key] = str(secrets[key])
        except Exception:  # noqa: BLE001 - never break the app over secrets
            continue


_load_streamlit_secrets()

from agents.llm import llm_is_configured  # noqa: E402
from agents.orchestrator import AnalysisResult, Orchestrator  # noqa: E402
from config.settings import get_settings  # noqa: E402
from utils.formatting import (  # noqa: E402
    fmt_confidence,
    fmt_currency,
    fmt_pct,
    sentiment_emoji,
)
from utils.logger import ConversationLog, get_logger  # noqa: E402
from utils.parsing import InputValidationError  # noqa: E402

logger = get_logger(__name__)
SETTINGS = get_settings()

st.set_page_config(
    page_title="Financial Analysis Agent Crew",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

CUSTOM_CSS = """
<style>
    .block-container {padding-top: 1.6rem; padding-bottom: 2rem; max-width: 1400px;}
    .crew-hero {
        background: linear-gradient(120deg, #0B3D91 0%, #0B6E4F 100%);
        padding: 1.4rem 1.6rem; border-radius: 14px; color: #fff; margin-bottom: 1.2rem;
    }
    .crew-hero h1 {color: #fff; margin: 0 0 .3rem 0; font-size: 1.85rem;}
    .crew-hero p {margin: 0; opacity: .92; font-size: .95rem;}
    .crew-card {
        border: 1px solid #E5E7EB; border-radius: 12px; padding: 1rem 1.1rem;
        background: #FFFFFF; height: 100%;
    }
    .crew-card h4 {margin: 0 0 .35rem 0; font-size: .78rem; letter-spacing: .06em;
        text-transform: uppercase; color: #6B7280;}
    .crew-card .value {font-size: 1.5rem; font-weight: 700; color: #111827;}
    .crew-card .note {font-size: .8rem; color: #6B7280; margin-top: .2rem;}
    .badge-pos {background:#E7F5EE;color:#0B6E4F;padding:.15rem .5rem;border-radius:999px;
        font-size:.78rem;font-weight:600;}
    .badge-neg {background:#FDECEA;color:#C0392B;padding:.15rem .5rem;border-radius:999px;
        font-size:.78rem;font-weight:600;}
    .badge-neu {background:#FEF7E0;color:#8A6D00;padding:.15rem .5rem;border-radius:999px;
        font-size:.78rem;font-weight:600;}
    .stTabs [data-baseweb="tab-list"] {gap: .35rem;}
    footer {visibility: hidden;}
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

QUICK_TICKERS = ["AAPL", "MSFT", "NVDA", "TSLA", "AMZN", "GOOGL", "META", "^GSPC"]
DATE_PRESETS = {
    "Last 30 days": 30,
    "Last 90 days": 90,
    "Last 6 months": 182,
    "Last 1 year": 365,
    "Last 2 years": 730,
}

AGENTS = [
    ("🔎", "Market Researcher", "News, themes and sentiment"),
    ("📊", "Data Analyst", "KPIs, risk metrics and charts"),
    ("✍️", "Report Writer", "Final report and recommendation"),
]


# ---------------------------------------------------------------------------
# Session state
# ---------------------------------------------------------------------------
def init_state() -> None:
    """Initialise the keys this app relies on."""
    st.session_state.setdefault("result", None)
    st.session_state.setdefault("history", [])
    st.session_state.setdefault("pdf_bytes", None)
    st.session_state.setdefault("error", None)


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
def render_sidebar() -> dict[str, object]:
    """Render configuration controls and return the chosen options."""
    with st.sidebar:
        st.markdown("### ⚙️ Configuration")
        configured = llm_is_configured()
        if configured:
            st.success(f"Gemini connected · `{SETTINGS.gemini_model}`", icon="✅")
        else:
            st.warning(
                "No `GOOGLE_API_KEY` found. The app will use the **deterministic offline "
                "crew** — every section, chart and PDF still works, but no LLM reasoning.",
                icon="🔑",
            )
            st.caption("Add `GOOGLE_API_KEY=...` to your `.env` file to enable the live agents.")

        use_llm = st.toggle(
            "Use live Gemini agents",
            value=configured,
            disabled=not configured,
            help="Turn off to force the offline crew (fast, free, deterministic).",
        )

        st.markdown("---")
        st.markdown("### 🗓️ Date presets")
        preset = st.radio(
            "Quick range",
            options=["Custom"] + list(DATE_PRESETS),
            index=2,
            label_visibility="collapsed",
        )

        st.markdown("---")
        st.markdown("### 🧠 Crew settings")
        revision_rounds = st.slider(
            "Max revision rounds",
            min_value=0,
            max_value=SETTINGS.max_revision_rounds,
            value=SETTINGS.max_revision_rounds,
            help="How many times the orchestrator can send the draft back to the Report Writer.",
        )
        include_log = st.toggle("Include conversation log in PDF", value=True)
        demo_toggle = st.toggle(
            "Force synthetic demo data",
            value=SETTINGS.force_demo_data,
            help="Skip live providers entirely and generate a labelled demo report.",
        )

        st.markdown("---")
        st.markdown("### ℹ️ About")
        st.caption(
            f"**{SETTINGS.app_name}** v{SETTINGS.app_version}\n\n"
            "Three specialised agents coordinated by an orchestrator, built with "
            "**agno**, **Gemini**, **yfinance**, **Plotly** and **reportlab**.\n\n"
            "Outputs are written to `reports/output/`."
        )
        with st.expander("Runtime details"):
            st.json(SETTINGS.public_dict())

        if st.session_state["history"]:
            st.markdown("---")
            st.markdown("### 🕘 Recent runs")
            for entry in reversed(st.session_state["history"][-5:]):
                st.caption(
                    f"**{entry['ticker']}** · {entry['start']} → {entry['end']} · "
                    f"{entry['recommendation'] or 'n/a'} · {entry['duration']}s"
                )

    return {
        "use_llm": use_llm,
        "preset": preset,
        "revision_rounds": revision_rounds,
        "include_log": include_log,
        "force_demo": demo_toggle,
    }


# ---------------------------------------------------------------------------
# Header & inputs
# ---------------------------------------------------------------------------
def render_header() -> None:
    st.markdown(
        """
        <div class="crew-hero">
            <h1>📈 Financial Analysis Agent Crew</h1>
            <p>Market Researcher → Data Analyst → Report Writer, coordinated by an orchestrator
            with a revision loop. News, sentiment, KPIs, charts and a downloadable PDF report.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_agent_row() -> None:
    columns = st.columns(len(AGENTS))
    for column, (icon, name, description) in zip(columns, AGENTS, strict=False):
        with column:
            st.markdown(
                f'<div class="crew-card"><h4>{icon} {name}</h4>'
                f'<div class="note">{description}</div></div>',
                unsafe_allow_html=True,
            )
    st.write("")


def render_inputs(options: dict[str, object]) -> tuple[str, date, date, bool]:
    """Render the analysis form and return its values."""
    preset = str(options["preset"])
    today = date.today()
    if preset in DATE_PRESETS:
        default_start = today - timedelta(days=DATE_PRESETS[preset])
    else:
        default_start = today - timedelta(days=182)

    with st.form("analysis_form", clear_on_submit=False):
        col_ticker, col_start, col_end, col_button = st.columns([1.3, 1, 1, 0.9])
        with col_ticker:
            ticker = (
                st.text_input(
                    "Stock ticker",
                    value="AAPL",
                    key="ticker_input",
                    placeholder="e.g. AAPL, TSLA, 0700.HK",
                    help="Yahoo Finance symbol. Indices work too, e.g. ^GSPC.",
                )
                .upper()
                .strip()
            )
            st.caption("Popular: " + " · ".join(QUICK_TICKERS[:5]))
        with col_start:
            start = st.date_input("From", value=default_start, max_value=today)
        with col_end:
            end = st.date_input("To", value=today, max_value=today)
        with col_button:
            st.write("")
            st.write("")
            submitted = st.form_submit_button("🚀 Analyze", width="stretch", type="primary")

        st.caption(
            "Analysis runs synchronously and typically takes 20-90 seconds with the live crew."
        )

    return ticker, start, end, submitted


# ---------------------------------------------------------------------------
# Progress
# ---------------------------------------------------------------------------
def make_progress_callback(status_holder: dict[str, object]):
    """Create a Streamlit progress callback for the orchestrator."""
    progress_bar = st.progress(0.0, text="Starting…")
    status_box = st.empty()
    status_holder["bar"] = progress_bar
    status_holder["box"] = status_box

    stage_icons = {
        "Input": "🧾",
        "Data": "📥",
        "Charts": "📈",
        "Agents": "🤖",
        "Revision": "🔁",
        "Review": "🧐",
        "Files": "🗂️",
        "PDF": "📄",
        "Done": "✅",
    }

    def callback(stage: str, fraction: float, message: str) -> None:
        icon = stage_icons.get(stage, "⏳")
        try:
            progress_bar.progress(min(1.0, max(0.0, fraction)), text=f"{icon} {stage}: {message}")
            status_box.caption(f"Latest update · {message}")
        except Exception:  # noqa: BLE001 - progress must never break a run
            pass

    return callback, progress_bar, status_box


def clear_progress(progress_bar, status_box) -> None:
    try:
        progress_bar.empty()
        status_box.empty()
    except Exception:  # noqa: BLE001
        pass


# ---------------------------------------------------------------------------
# Results
# ---------------------------------------------------------------------------
def render_metric_cards(result: AnalysisResult) -> None:
    kpis = result.kpis
    cards = [
        ("Last price", fmt_currency(kpis.get("current_price")), kpis.get("company_name", "")),
        (
            "Period ROI",
            fmt_pct(kpis.get("period_roi_pct")),
            f"{kpis.get('trading_days')} trading days",
        ),
        (
            "Sentiment",
            f"{sentiment_emoji(result.sentiment.get('label'))} {str(result.sentiment.get('label', 'n/a')).title()}",
            f"score {result.sentiment.get('score', 0):+.2f} · {result.sentiment.get('articles_analyzed', 0)} headlines",
        ),
        (
            "Recommendation",
            result.recommendation or "n/a",
            f"confidence {fmt_confidence(result.confidence)}",
        ),
        ("Run time", f"{result.duration_s:.1f}s", f"{result.revision_rounds} revision round(s)"),
    ]
    columns = st.columns(len(cards))
    for column, (label, value, note) in zip(columns, cards, strict=False):
        with column:
            st.markdown(
                f'<div class="crew-card"><h4>{label}</h4>'
                f'<div class="value">{value}</div>'
                f'<div class="note">{note}</div></div>',
                unsafe_allow_html=True,
            )
    st.write("")


def render_status_banner(result: AnalysisResult) -> None:
    if result.is_demo:
        st.warning(
            "**Synthetic demo data.** Live market/news providers were unreachable or demo mode "
            "was forced. Figures are simulated for demonstration only.",
            icon="🧪",
        )
    if not result.llm_used:
        st.info(
            "This run used the **deterministic offline crew** (no LLM). The report, charts and "
            "PDF are complete; narrative quality improves with a Gemini API key.",
            icon="🤖",
        )
    if result.warnings:
        with st.expander(f"⚠️ Run notes ({len(result.warnings)})", expanded=False):
            for warning in result.warnings:
                st.markdown(f"- {warning}")


def render_downloads(result: AnalysisResult, include_log: bool) -> None:
    columns = st.columns([1, 1, 1, 1])
    with columns[0]:
        pdf_bytes = st.session_state.get("pdf_bytes")
        if pdf_bytes:
            st.download_button(
                "📄 Download PDF report",
                data=pdf_bytes,
                file_name=os.path.basename(result.pdf_path or f"{result.ticker}_report.pdf"),
                mime="application/pdf",
                width="stretch",
                type="primary",
            )
        else:
            st.button("📄 PDF unavailable", disabled=True, width="stretch")
    with columns[1]:
        st.download_button(
            "📝 Download Markdown",
            data=result.report_markdown.encode("utf-8"),
            file_name=f"{result.ticker}_report.md",
            mime="text/markdown",
            width="stretch",
        )
    with columns[2]:
        import json

        st.download_button(
            "🧾 Download KPI JSON",
            data=json.dumps(
                {"kpis": result.kpis, "sentiment": result.sentiment}, indent=2, default=str
            ).encode("utf-8"),
            file_name=f"{result.ticker}_kpis.json",
            mime="application/json",
            width="stretch",
        )
    with columns[3]:
        import json

        st.download_button(
            "🗣️ Download agent log",
            data=json.dumps(result.conversation, indent=2, default=str).encode("utf-8"),
            file_name=f"{result.ticker}_conversation.json",
            mime="application/json",
            width="stretch",
        )


def render_results(result: AnalysisResult, options: dict[str, object]) -> None:
    render_metric_cards(result)
    render_status_banner(result)
    render_downloads(result, bool(options["include_log"]))

    tabs = st.tabs(
        [
            "📄 Report",
            "📈 Charts",
            "📊 KPI Table",
            "📰 News & Sentiment",
            "🗣️ Agent Conversation Log",
            "🧾 Raw data",
        ]
    )

    with tabs[0]:
        st.markdown(result.report_markdown)
        if result.pdf_path:
            st.caption(f"PDF saved to `{result.pdf_path}`")

    with tabs[1]:
        if not result.figures:
            st.info("No interactive charts were produced for this run.")
        else:
            labels = {
                "price": "Price & moving averages",
                "volume": "Volume",
                "rsi": "RSI (14)",
                "kpi": "KPI overview",
            }
            for key, figure in result.figures.items():
                st.markdown(f"#### {labels.get(key, key.title())}")
                st.plotly_chart(figure, width="stretch", key=f"chart_{key}")
        if result.chart_images:
            with st.expander("Static images used in the PDF"):
                st.caption(", ".join(os.path.basename(p) for p in result.chart_images.values()))

    with tabs[2]:
        from tools.finance_tools import build_kpi_table

        rows = build_kpi_table(result.kpis)
        st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)
        st.caption(
            f"Source: {result.kpis.get('source')} · window {result.start} → {result.end} · "
            f"{result.kpis.get('trading_days')} trading days"
        )

    with tabs[3]:
        sentiment = result.sentiment
        col_a, col_b, col_c, col_d = st.columns(4)
        col_a.metric("Aggregate", str(sentiment.get("label", "n/a")).title())
        col_b.metric("Score", f"{sentiment.get('score', 0):+.2f}")
        col_c.metric(
            "Positive / Negative",
            f"{sentiment.get('positive_count', 0)} / {sentiment.get('negative_count', 0)}",
        )
        col_d.metric("Headlines", sentiment.get("articles_analyzed", 0))

        if sentiment.get("observations"):
            for observation in sentiment["observations"]:
                st.markdown(f"- {observation}")

        st.markdown("#### Headlines")
        if not result.news:
            st.info("No headlines were retrieved for this window.")
        for item in result.news:
            title = item.get("title", "(untitled)")
            meta = " · ".join(part for part in (item.get("source"), item.get("published")) if part)
            if item.get("url"):
                st.markdown(f"**[{title}]({item['url']})**  \n{meta}")
            else:
                st.markdown(f"**{title}**  \n{meta}")
            if item.get("summary"):
                st.caption(item["summary"])

        scored = sentiment.get("items") or []
        if scored:
            with st.expander("Per-headline sentiment scores"):
                frame = pd.DataFrame(scored)[["title", "source", "published", "score", "label"]]
                st.dataframe(frame, width="stretch", hide_index=True)

    with tabs[4]:
        st.caption(
            "Every agent hand-off, tool call and revision round is recorded below. "
            "The same log is appended to the PDF when enabled."
        )
        for event in result.conversation:
            st.markdown(
                f"- `{event.get('timestamp', '')}` **{event.get('agent', '')}** "
                f"_{event.get('event', '')}_: {event.get('message', '')}"
            )

    with tabs[5]:
        with st.expander("KPI dictionary"):
            st.json(result.kpis)
        with st.expander("Run timings"):
            st.json(result.timings)
        if result.report_path:
            st.caption(f"Markdown saved to `{result.report_path}`")
        if result.log_path:
            st.caption(f"Conversation log saved to `{result.log_path}`")


def render_empty_state() -> None:
    st.markdown("#### How it works")
    steps = st.columns(3)
    descriptions = [
        (
            "1 · Collect",
            "The Market Researcher pulls recent headlines and a finance-tuned sentiment "
            "model scores each one. The Data Analyst downloads price history and computes "
            "ROI, volatility, Sharpe, drawdown, RSI, moving averages and valuation ratios.",
        ),
        (
            "2 · Reason",
            "The orchestrator (an agno Team in coordinate mode) passes findings downstream. "
            "The Report Writer drafts the report, and the orchestrator can send it back once "
            "or twice with specific revision instructions.",
        ),
        (
            "3 · Deliver",
            "Plotly charts are rendered to PNG and embedded, with the KPI table, sentiment, "
            "recommendation, confidence score and disclaimer, into a reportlab PDF you can "
            "download from this page.",
        ),
    ]
    for column, (title, body) in zip(steps, descriptions, strict=False):
        with column:
            st.markdown(
                f'<div class="crew-card"><h4>{title}</h4><div class="note">{body}</div></div>',
                unsafe_allow_html=True,
            )
    st.write("")
    st.info("Enter a ticker and date range above, then press **Analyze**.", icon="👉")
    if SETTINGS.force_demo_data:
        st.caption("Demo mode is enabled, so any ticker will produce a labelled synthetic report.")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def run_and_store(ticker: str, start: date, end: date, options: dict[str, object]) -> None:
    """Execute the analysis with live progress and store the result."""
    holder: dict[str, object] = {}
    callback, progress_bar, status_box = make_progress_callback(holder)
    log = ConversationLog()

    orchestrator = Orchestrator(
        progress=callback,
        log=log,
        generate_pdf=True,
        max_revision_rounds=int(options["revision_rounds"]),
        force_demo=bool(options["force_demo"]),
        pdf_include_conversation=bool(options["include_log"]),
    )
    try:
        result = orchestrator.run(ticker, start, end, use_llm=bool(options["use_llm"]))
    finally:
        clear_progress(progress_bar, status_box)

    st.session_state["result"] = result
    st.session_state["error"] = None
    st.session_state["pdf_bytes"] = None
    if result.pdf_path and os.path.exists(result.pdf_path):
        try:
            with open(result.pdf_path, "rb") as handle:
                st.session_state["pdf_bytes"] = handle.read()
        except OSError:
            logger.warning("Could not read generated PDF %s", result.pdf_path)

    st.session_state["history"].append(
        {
            "ticker": result.ticker,
            "start": str(result.start),
            "end": str(result.end),
            "recommendation": result.recommendation,
            "duration": result.duration_s,
        }
    )
    st.session_state["history"] = st.session_state["history"][-20:]


def main() -> None:
    init_state()
    render_header()
    options = render_sidebar()
    render_agent_row()
    ticker, start, end, submitted = render_inputs(options)

    if submitted:
        st.session_state["error"] = None
        st.session_state["pdf_bytes"] = None
        if not ticker:
            st.error("Please enter a stock ticker.")
        elif start >= end:
            st.error("The start date must be before the end date.")
        else:
            try:
                with st.status(f"Analysing **{ticker}**…", expanded=True) as status:
                    st.write("Fetching market data and news…")
                    run_and_store(ticker, start, end, options)
                    status.update(label=f"Analysis complete for {ticker}", state="complete")
            except InputValidationError as exc:
                st.session_state["error"] = str(exc)
                st.error(str(exc), icon="⚠️")
            except Exception as exc:  # noqa: BLE001 - surface a readable message
                logger.exception("Analysis failed")
                st.session_state["error"] = f"{type(exc).__name__}: {exc}"
                st.error(
                    f"Analysis failed: {type(exc).__name__}: {exc}\n\n"
                    "Try a different ticker, a shorter date range, or enable demo mode "
                    "in the sidebar.",
                    icon="🚨",
                )

    result: AnalysisResult | None = st.session_state.get("result")
    if result is not None:
        st.markdown("---")
        st.markdown(
            f"## Results · {result.company_name} ({result.ticker})  "
            f"<span class='badge-{'pos' if result.recommendation in {'BUY', 'STRONG BUY'} else 'neg' if result.recommendation in {'SELL', 'STRONG SELL'} else 'neu'}'>"
            f"{result.recommendation or 'N/A'}</span>",
            unsafe_allow_html=True,
        )
        render_results(result, options)
    else:
        render_empty_state()

    st.markdown("---")
    st.caption(
        "⚠️ This tool produces automated research for educational purposes only. It is not "
        "investment advice. Data may be delayed, incomplete or synthetic — always verify "
        "figures independently."
    )


if __name__ == "__main__":
    main()
