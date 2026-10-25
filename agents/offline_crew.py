"""Deterministic, no-API-key crew used when Gemini is unavailable.

This module is the safety net for the whole application:

* With no ``GOOGLE_API_KEY`` (or when the API fails) the orchestrator still needs
  a complete, honest report. Here we build the same three artefacts a live run
  would produce — research brief, analyst brief and final report — directly from
  the tool outputs.
* It doubles as the section-filler for LLM reports that omit a required heading,
  so the PDF never ships half-empty.

Nothing here calls an LLM, so it is fast, free, deterministic and fully unit
tested. Every report it produces is labelled as system-generated.
"""

from __future__ import annotations

from collections.abc import Iterable
from datetime import date
from typing import Any

from prompts.prompts import DISCLAIMER_TEXT
from utils.formatting import (
    fmt_compact,
    fmt_currency,
    fmt_number,
    fmt_pct,
    sentiment_emoji,
)


def _as_float(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return None if number != number else number


def heuristic_recommendation(
    kpis: dict[str, Any], sentiment: dict[str, Any] | None = None
) -> dict[str, Any]:
    """Derive a transparent, rule-based recommendation and confidence score.

    The score is a weighted sum of momentum, trend, risk and sentiment signals.
    It is intentionally simple and explainable, and it is only used when no LLM
    verdict is available (or as a fallback when parsing fails).

    Returns:
        Dict with ``recommendation``, ``confidence``, ``rationale`` and ``score``.
    """
    sentiment = sentiment or {}
    score = 0.0
    drivers: list[str] = []

    roi = _as_float(kpis.get("period_roi_pct"))
    relative = _as_float(kpis.get("relative_return_pct"))
    momentum_20 = _as_float(kpis.get("momentum_20d_pct"))
    rsi = _as_float(kpis.get("rsi_14"))
    sharpe = _as_float(kpis.get("sharpe_ratio"))
    drawdown = _as_float(kpis.get("max_drawdown_pct"))
    volatility = _as_float(kpis.get("volatility_annual_pct"))
    price_vs_sma50 = _as_float(kpis.get("price_vs_sma50_pct"))
    pe = _as_float(kpis.get("pe_ratio"))
    sentiment_score = _as_float(sentiment.get("score"))

    if roi is not None:
        contribution = max(-2.0, min(2.0, roi / 10))
        score += contribution
        drivers.append(f"period ROI of {fmt_pct(roi)} ({contribution:+.2f})")
    if relative is not None:
        contribution = max(-1.5, min(1.5, relative / 8))
        score += contribution
        drivers.append(
            f"relative performance vs benchmark of {fmt_pct(relative)} ({contribution:+.2f})"
        )
    if momentum_20 is not None:
        contribution = max(-1.5, min(1.5, momentum_20 / 8))
        score += contribution
        drivers.append(f"20-day momentum of {fmt_pct(momentum_20)} ({contribution:+.2f})")
    if price_vs_sma50 is not None:
        contribution = max(-1.5, min(1.5, price_vs_sma50 / 8))
        score += contribution
        drivers.append(f"price {fmt_pct(price_vs_sma50)} vs SMA50 ({contribution:+.2f})")
    if rsi is not None:
        if rsi >= 75:
            score -= 0.8
            drivers.append(f"RSI {rsi:.1f} signals overbought conditions (-0.80)")
        elif rsi <= 25:
            score += 0.4
            drivers.append(f"RSI {rsi:.1f} signals oversold conditions (+0.40)")
        else:
            drivers.append(f"RSI {rsi:.1f} is neutral")
    if sharpe is not None:
        contribution = max(-1.0, min(1.0, sharpe / 2))
        score += contribution
        drivers.append(f"Sharpe ratio of {fmt_number(sharpe)} ({contribution:+.2f})")
    if drawdown is not None and drawdown > 25:
        score -= 0.5
        drivers.append(f"max drawdown of {fmt_pct(drawdown, signed=False)} (-0.50)")
    if volatility is not None and volatility > 55:
        score -= 0.5
        drivers.append(f"elevated volatility of {fmt_pct(volatility, signed=False)} (-0.50)")
    if pe is not None:
        if pe > 45:
            score -= 0.6
            drivers.append(f"rich valuation at {fmt_number(pe)}x earnings (-0.60)")
        elif 0 < pe < 15:
            score += 0.5
            drivers.append(f"undemanding valuation at {fmt_number(pe)}x earnings (+0.50)")
    if sentiment_score is not None:
        contribution = max(-1.5, min(1.5, sentiment_score * 2.5))
        score += contribution
        label = sentiment.get("label", "neutral")
        drivers.append(f"{label} news sentiment ({contribution:+.2f})")

    if score >= 2.5:
        recommendation = "STRONG BUY"
    elif score >= 1.0:
        recommendation = "BUY"
    elif score <= -2.5:
        recommendation = "STRONG SELL"
    elif score <= -1.0:
        recommendation = "SELL"
    else:
        recommendation = "HOLD"

    # Confidence blends data completeness, signal strength and agreement between
    # the quantitative and qualitative reads. Capped at 85 (and much lower for
    # synthetic data) because this is a heuristic, not a backtest.
    available = (roi, relative, momentum_20, rsi, sharpe, drawdown, pe, sentiment_score)
    filled = sum(1 for value in available if value is not None)
    strength = min(1.0, abs(score) / 3.0)
    confidence = 38.0 + filled * 4.0 + 8.0 * strength

    if sentiment_score is not None and abs(score) > 0.3 and (score > 0) != (sentiment_score > 0):
        confidence -= 6.0
        drivers.append("quantitative and news signals disagree (-6 confidence)")
    if kpis.get("is_demo"):
        confidence -= 18.0
    trading_days = _as_float(kpis.get("trading_days")) or 0
    if trading_days < 60:
        confidence -= 8.0
    confidence_int = int(max(20, min(85, round(confidence))))

    rationale = (
        "Rule-based confidence from data completeness and signal agreement. "
        f"Signals used: {'; '.join(drivers[:6]) or 'none available'}."
    )
    return {
        "recommendation": recommendation,
        "confidence": confidence_int,
        "rationale": rationale,
        "score": round(score, 2),
        "drivers": drivers,
        "source": "heuristic",
    }


def research_brief(
    news: Iterable[dict[str, Any]],
    sentiment: dict[str, Any],
    ticker: str,
    company_name: str,
) -> str:
    """Build the Market Researcher's Markdown brief from tool output."""
    news = list(news or [])
    label = sentiment.get("label", "neutral")
    score = sentiment.get("score", 0.0)

    lines = ["## News Summary"]
    if not news:
        lines.append("- No headlines could be retrieved for this window.")
    else:
        for item in news[:6]:
            published = f" ({item['published']})" if item.get("published") else ""
            source = f" — {item['source']}" if item.get("source") else ""
            demo = " [synthetic]" if item.get("is_demo") else ""
            lines.append(f"- {item.get('title', '').strip()}{source}{published}{demo}")

    lines += [
        "",
        "## Sentiment",
        (
            f"{sentiment_emoji(label)} Aggregate sentiment is **{label}** "
            f"(score {score:+.2f} on a −1 to +1 scale) across "
            f"{sentiment.get('articles_analyzed', 0)} headlines: "
            f"{sentiment.get('positive_count', 0)} positive, "
            f"{sentiment.get('neutral_count', 0)} neutral, "
            f"{sentiment.get('negative_count', 0)} negative."
        ),
    ]
    observations = sentiment.get("observations") or []
    if observations:
        lines.append("")
        lines.extend(f"- {observation}" for observation in observations)

    lines += ["", "## Key Drivers"]
    drivers = _news_drivers(sentiment, news)
    lines.extend(f"- {driver}" for driver in drivers) if drivers else lines.append(
        "- No company-specific drivers identified from available sources."
    )

    lines += ["", "## Risks & Gaps"]
    gaps = [
        "Headline coverage may miss paywalled or non-English reporting.",
        "Sentiment is lexicon-based and cannot read sarcasm or nuanced guidance language.",
    ]
    if any(item.get("is_demo") for item in news):
        gaps.insert(0, "SYNTHETIC DEMO DATA: headlines above are simulated, not real news.")
    if not news:
        gaps.insert(0, "No news items were available, so sentiment defaults to neutral.")
    lines.extend(f"- {gap}" for gap in gaps)
    return "\n".join(lines)


def _news_drivers(sentiment: dict[str, Any], news: list[dict[str, Any]]) -> list[str]:
    """Derive short driver statements from scored headlines."""
    items = sentiment.get("items") or []
    drivers: list[str] = []
    positives = [item for item in items if item.get("label") == "positive"]
    negatives = [item for item in items if item.get("label") == "negative"]

    for item in positives[:2]:
        drivers.append(f"[positive] {item['title']}")
    for item in negatives[:2]:
        drivers.append(f"[negative] {item['title']}")
    if not positives and not negatives:
        drivers.append("[neutral] Headline tone is balanced with no dominant catalyst.")
    if len(news) < 3:
        drivers.append("[neutral] Thin news flow limits conviction in the sentiment read.")
    return drivers


def analyst_brief(kpis: dict[str, Any], ticker: str, company_name: str) -> str:
    """Build the Data Analyst's Markdown brief from computed KPIs."""
    from tools.finance_tools import build_kpi_table

    currency = "$" if kpis.get("currency", "USD") == "USD" else f"{kpis.get('currency')} "
    rows = build_kpi_table(kpis)
    interesting = [
        "Current price",
        "Period return (ROI)",
        "Benchmark return",
        "Relative return",
        "Volatility (annual)",
        "Sharpe ratio",
        "Max drawdown",
        "RSI (14)",
        "SMA 50",
        "Trend signal",
        "P/E (trailing)",
        "Market cap",
    ]
    selected = [row for row in rows if row["metric"] in interesting]

    summary = (
        f"{company_name} ({ticker}) closed at "
        f"{fmt_currency(kpis.get('current_price'), symbol=currency)} on {kpis.get('end')}, "
        f"a {fmt_pct(kpis.get('period_roi_pct'))} move over the window. "
        f"Annualised volatility is {fmt_pct(kpis.get('volatility_annual_pct'), signed=False)} "
        f"with a Sharpe ratio of {fmt_number(kpis.get('sharpe_ratio'))} and a maximum drawdown "
        f"of {fmt_pct(kpis.get('max_drawdown_pct'), signed=False)}. "
        f"The moving-average structure reads: {kpis.get('ma_trend')}."
    )

    lines = [
        "## Quantitative Assessment",
        summary,
        "",
        "## KPI Highlights",
        "| Metric | Value | Read |",
        "| --- | --- | --- |",
    ]
    for row in selected:
        lines.append(f"| {row['metric']} | {row['value']} | {row['note']} |")

    lines += ["", "## Risk Profile"]
    risks = []
    volatility = _as_float(kpis.get("volatility_annual_pct"))
    if volatility is not None:
        level = "high" if volatility > 45 else "moderate" if volatility > 25 else "low"
        risks.append(
            f"{level.capitalize()} volatility ({fmt_pct(volatility, signed=False)} annualised)."
        )
    beta = _as_float(kpis.get("beta"))
    if beta is not None:
        risks.append(
            f"Beta of {fmt_number(beta)} implies "
            f"{'above' if beta > 1 else 'below'}-market sensitivity."
        )
    if kpis.get("max_drawdown_pct") is not None:
        risks.append(
            f"Worst peak-to-trough drawdown in the window: {fmt_pct(kpis.get('max_drawdown_pct'), signed=False)}."
        )
    if kpis.get("avg_volume") is not None:
        risks.append(
            f"Average volume of {fmt_compact(kpis.get('avg_volume'), symbol='')} shares supports liquidity."
        )
    if not risks:
        risks.append("Insufficient history to characterise risk.")
    lines.extend(f"- {risk}" for risk in risks)

    lines += ["", "## Data Quality"]
    quality = list(kpis.get("data_quality") or [])
    lines.extend(f"- {note}" for note in quality) if quality else lines.append(
        "- Full data coverage for the selected window."
    )

    verdict = heuristic_recommendation(kpis)
    lines += [
        "",
        "## Analyst Verdict",
        (
            f"Rule-based read: **{verdict['recommendation']}** (signal score "
            f"{verdict['score']:+.2f}). Strongest support: "
            f"{fmt_pct(kpis.get('period_roi_pct'))} period return with a "
            f"{fmt_number(kpis.get('sharpe_ratio'))} Sharpe ratio. Key caution: "
            f"{fmt_pct(kpis.get('volatility_annual_pct'), signed=False)} volatility and a "
            f"{fmt_pct(kpis.get('max_drawdown_pct'), signed=False)} drawdown."
        ),
    ]
    return "\n".join(lines)


def compose_report(
    *,
    ticker: str,
    company_name: str,
    kpis: dict[str, Any],
    sentiment: dict[str, Any],
    news: list[dict[str, Any]],
    start: date,
    end: date,
    revision_note: str | None = None,
) -> str:
    """Compose the complete Markdown report without an LLM."""
    from tools.finance_tools import build_kpi_table

    verdict = heuristic_recommendation(kpis, sentiment)
    currency = "$" if kpis.get("currency", "USD") == "USD" else f"{kpis.get('currency')} "
    label = sentiment.get("label", "neutral")
    rows = build_kpi_table(kpis)

    exec_summary = (
        f"{company_name} ({ticker}) was analysed over {start} → {end}. The shares moved "
        f"{fmt_pct(kpis.get('period_roi_pct'))} to {fmt_currency(kpis.get('current_price'), symbol=currency)}, "
        f"{_relative_phrase(kpis)}, while news sentiment is {label} "
        f"(score {sentiment.get('score', 0.0):+.2f}). Risk is "
        f"{'high' if (_as_float(kpis.get('volatility_annual_pct')) or 0) > 45 else 'moderate'}, "
        f"with {fmt_pct(kpis.get('volatility_annual_pct'), signed=False)} annualised volatility and a "
        f"{fmt_pct(kpis.get('max_drawdown_pct'), signed=False)} maximum drawdown. "
        f"On this evidence the systematic view is **{verdict['recommendation']}** with "
        f"{verdict['confidence']}/100 confidence. "
        + ("⚠️ Figures are synthetic demo data. " if kpis.get("is_demo") else "")
    )

    snapshot = (
        f"{company_name} trades on {kpis.get('exchange') or 'an unspecified exchange'} in the "
        f"{kpis.get('sector') or 'unknown'} sector ({kpis.get('industry') or 'n/a'}). "
        f"Market capitalisation is {fmt_compact(kpis.get('market_cap'), symbol=currency)} and the "
        f"stock sits at {fmt_number(kpis.get('range_position_pct'), 0)}% of its 52-week range "
        f"({fmt_currency(kpis.get('low_52w'), symbol=currency)} – "
        f"{fmt_currency(kpis.get('high_52w'), symbol=currency)}). P/E is "
        f"{fmt_number(kpis.get('pe_ratio'))}x with EPS of {fmt_currency(kpis.get('eps'), symbol=currency)}."
    )

    news_block = research_brief(news, sentiment, ticker, company_name)

    kpi_lines = ["| Metric | Value | Interpretation |", "| --- | --- | --- |"]
    for row in rows:
        kpi_lines.append(f"| {row['metric']} | {row['value']} | {row['note']} |")

    chart_notes = [
        "- **Price with moving averages** — shows the trend structure; price above SMA50 with a "
        "rising SMA20 indicates positive momentum.",
        "- **Trading volume** — unusually high volume around price moves increases the reliability "
        "of the signal.",
        "- **RSI (14)** — values above 70 flag overbought conditions, below 30 oversold.",
        "- **KPI bar chart** — compares return, volatility, drawdown, RSI, P/E, Sharpe and beta on "
        "one axis for a quick visual risk/return check.",
    ]

    recommendation_lead = (
        f"the shares returned {fmt_pct(kpis.get('period_roi_pct'))} over the window, "
        f"20-day momentum is {fmt_pct(kpis.get('momentum_20d_pct'))}, the trend reads "
        f'"{kpis.get("ma_trend")}" and news sentiment is {label} '
        f"({sentiment.get('score', 0.0):+.2f}); together these give a systematic signal "
        f"score of {verdict['score']:+.2f}."
    )

    risk_lines = [
        f"- **Market risk:** {fmt_pct(kpis.get('volatility_annual_pct'), signed=False)} annualised "
        f"volatility and a beta of {fmt_number(kpis.get('beta'))}.",
        f"- **Drawdown risk:** the worst peak-to-trough decline in the window was "
        f"{fmt_pct(kpis.get('max_drawdown_pct'), signed=False)}.",
        f"- **Valuation risk:** trailing P/E of {fmt_number(kpis.get('pe_ratio'))}x "
        f"{'is elevated and vulnerable to earnings disappointment' if (_as_float(kpis.get('pe_ratio')) or 0) > 35 else 'looks broadly reasonable for the sector'}.",
        f"- **News risk:** aggregate sentiment is {label}; a shift in tone could pressure the shares.",
        "- **Data risk:** "
        + (
            "this run uses synthetic demo data and must not inform real decisions."
            if kpis.get("is_demo")
            else "provider data may be delayed or incomplete; verify against filings."
        ),
    ]

    demo_confidence_note = (
        "- Synthetic demo data caps confidence below 70.\n" if kpis.get("is_demo") else ""
    )
    demo_banner = (
        "\n> ⚠️ **SYNTHETIC DEMO DATA** — figures are simulated, not market data.\n"
        if kpis.get("is_demo")
        else ""
    )

    report = f"""# {ticker} — Financial Analysis Report

> **Generated by the Financial Analysis Agent Crew** on {date.today().isoformat()} · analysis window {start} → {end} · data source: {kpis.get("source", "yfinance")}
{demo_banner}
## Executive Summary
{exec_summary}

## Company Snapshot
{snapshot}

## News & Market Sentiment
{news_block.replace("## News Summary", "").replace("## Sentiment", "**Sentiment**").replace("## Key Drivers", "**Key drivers**").replace("## Risks & Gaps", "**Risks & gaps**").strip()}

## KPI Table
{chr(10).join(kpi_lines)}

## Charts
{chr(10).join(chart_notes)}

## Risk Assessment
{chr(10).join(risk_lines)}

## Recommendation
**{verdict["recommendation"]}** — {recommendation_lead}

Position risk to weigh alongside this view: {fmt_pct(kpis.get("volatility_annual_pct"), signed=False)}
annualised volatility, a {fmt_pct(kpis.get("max_drawdown_pct"), signed=False)} maximum drawdown and a
Sharpe ratio of {fmt_number(kpis.get("sharpe_ratio"))}. Return versus the benchmark is
{fmt_pct(kpis.get("relative_return_pct"))}.

## Confidence Score
**Confidence:** {verdict["confidence"]}/100 ({"moderate" if verdict["confidence"] >= 60 else "low"})

{verdict["rationale"]}
{demo_confidence_note}- Missing or short history reduces confidence.
- Sentiment is headline-based and backward-looking.

## Methodology & Limitations
Data was retrieved for {ticker} from {kpis.get("source", "yfinance")} for {start} → {end}
({kpis.get("trading_days", 0)} trading days). KPIs are computed from daily closes:
ROI, annualised return, annualised volatility (√{252} scaling), Sharpe ratio using a
{kpis.get("risk_free_rate_pct", 4.0)}% risk-free rate, maximum drawdown, Wilder RSI(14), SMA20/50/200
and momentum windows. Sentiment is derived from recent headlines using a finance-tuned
lexicon model. Charts are rendered with Plotly and rasterised for the PDF.

Limitations: no fundamental filings analysis, no options-implied volatility, no
intraday data, and news coverage may be incomplete. This is a systematic, rule-assisted
view — not a substitute for due diligence.

## Disclaimer
{DISCLAIMER_TEXT}
"""

    if revision_note:
        report += f"\n---\n_Revision note: {revision_note}_\n"
    return report.strip()


def _relative_phrase(kpis: dict[str, Any]) -> str:
    relative = _as_float(kpis.get("relative_return_pct"))
    benchmark = kpis.get("benchmark_ticker") or "the benchmark"
    if relative is None:
        return f"versus an unavailable {benchmark} benchmark"
    direction = "outperforming" if relative >= 0 else "underperforming"
    return f"{direction} {benchmark} by {abs(relative):.2f}pp"


__all__ = [
    "analyst_brief",
    "compose_report",
    "heuristic_recommendation",
    "research_brief",
]
