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
    """Build the Market Researcher's Markdown brief from tool output.

    Mirrors the headings the live agent is asked to produce:
    Key News Items, Sentiment, Key Risks, Catalysts (and Shortcomings when the
    news flow is thin or synthetic).
    """
    news = list(news or [])
    scored = sentiment.get("items") or []
    label = sentiment.get("label", "neutral")
    score = sentiment.get("score", 0.0)

    def _classification(index: int) -> str:
        """Positive/Neutral/Negative plus a short reason, per the prompt contract."""
        if index >= len(scored):
            return "**Neutral** — no sentiment score available for this item"
        item = scored[index]
        item_label = str(item.get("label", "neutral")).title()
        item_score = float(item.get("score", 0.0) or 0.0)
        reason = (
            "headline tone is constructive"
            if item_score > 0.15
            else "headline tone is negative"
            if item_score < -0.15
            else "headline tone is balanced"
        )
        return f"**{item_label}** — {reason} (score {item_score:+.2f})"

    lines = ["## Key News Items"]
    if not news:
        lines.append("- No headlines could be retrieved for this window.")
    else:
        for index, item in enumerate(news[:10]):
            published = item.get("published") or "date n/a"
            source = item.get("source") or "unknown source"
            demo = " [synthetic]" if item.get("is_demo") else ""
            lines.append(
                f"- {published} — {item.get('title', '').strip()} — {source}{demo} — "
                f"{_classification(index)}"
            )

    lines += [
        "",
        "## Sentiment",
        (
            f"{sentiment_emoji(label)} Overall sentiment score: **{score:+.2f}** on a -1 to "
            f"+1 scale (label: {label}) across {sentiment.get('articles_analyzed', 0)} "
            f"headlines — {sentiment.get('positive_count', 0)} positive, "
            f"{sentiment.get('neutral_count', 0)} neutral, "
            f"{sentiment.get('negative_count', 0)} negative."
        ),
    ]
    observations = sentiment.get("observations") or []
    if observations:
        lines.extend(f"- {observation}" for observation in observations)

    lines += ["", "## Key Risks"]
    risks = _news_risks(sentiment, news)
    lines.extend(f"- {risk}" for risk in risks)

    lines += ["", "## Catalysts"]
    catalysts = _news_catalysts(sentiment, news)
    lines.extend(f"- {catalyst}" for catalyst in catalysts)

    gaps = [_no_news_note(news)]
    if any(item.get("is_demo") for item in news):
        gaps.insert(0, "Headlines above are SYNTHETIC DEMO DATA, not real news.")
    gaps.append("Headline coverage may miss paywalled or non-English reporting.")
    if gaps:
        lines += ["", "## Shortcomings"]
        lines.extend(f"- {gap}" for gap in gaps)
    return "\n".join(lines)


def _no_news_note(news: list[dict[str, Any]]) -> str:
    if news:
        return "Social media / tweet coverage could not be retrieved; news is headline-only."
    return "No news items were available, so sentiment defaults to neutral."


def _news_risks(sentiment: dict[str, Any], news: list[dict[str, Any]]) -> list[str]:
    """Derive risk bullets from the most negative classified headlines."""
    scored = sentiment.get("items") or []
    negatives = [item for item in scored if item.get("label") == "negative"]
    risks = [f"Negative coverage: {item['title']}" for item in negatives[:2]]
    risks.append("Headline-driven sentiment is backward-looking and can reverse quickly.")
    if len(news) < 3:
        risks.append("Thin news flow: low information content limits conviction.")
    return risks


def _news_catalysts(sentiment: dict[str, Any], news: list[dict[str, Any]]) -> list[str]:
    """Derive catalyst bullets from the most positive classified headlines."""
    scored = sentiment.get("items") or []
    positives = [item for item in scored if item.get("label") == "positive"]
    catalysts = [f"Positive coverage to watch: {item['title']}" for item in positives[:2]]
    catalysts.append("Next earnings release and any guidance update.")
    catalysts.append("Analyst rating or price-target revisions.")
    return catalysts


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
    """Compose the complete Markdown report without an LLM.

    The structure matches the Report Writer contract exactly:
    Executive Summary, Market News Summary, KPI Table and Charts, Sentiment
    Overview, Analysis and Outlook, Recommendation (with the confidence score)
    and Disclaimer.
    """
    from tools.finance_tools import build_kpi_table

    verdict = heuristic_recommendation(kpis, sentiment)
    currency = "$" if kpis.get("currency", "USD") == "USD" else f"{kpis.get('currency')} "
    label = sentiment.get("label", "neutral")
    score = sentiment.get("score", 0.0)
    rows = build_kpi_table(kpis)

    volatility = _as_float(kpis.get("volatility_annual_pct")) or 0.0
    pe = _as_float(kpis.get("pe_ratio")) or 0.0
    confidence_label = (
        "high"
        if verdict["confidence"] >= 75
        else "moderate"
        if verdict["confidence"] >= 55
        else "low"
    )

    exec_summary = (
        f"**Conclusion: {verdict['recommendation']}** with {verdict['confidence']}/100 confidence "
        f"for {company_name} ({ticker}) over {start} -> {end}. The shares moved "
        f"{fmt_pct(kpis.get('period_roi_pct'))} to "
        f"{fmt_currency(kpis.get('current_price'), symbol=currency)}, "
        f"{_relative_phrase(kpis)}, with {fmt_pct(volatility, signed=False)} annualised volatility "
        f"and a {fmt_pct(kpis.get('max_drawdown_pct'), signed=False)} maximum drawdown. News "
        f"sentiment is {label} ({score:+.2f}) across {sentiment.get('articles_analyzed', 0)} "
        f'headlines, and the trend reads "{kpis.get("ma_trend")}". Valuation stands at '
        f"{fmt_number(kpis.get('pe_ratio'))}x trailing earnings on a "
        f"{fmt_compact(kpis.get('market_cap'), symbol=currency)} market cap. "
        + (
            "Figures are synthetic demo data and must not inform real decisions."
            if kpis.get("is_demo")
            else "The systematic view is detailed in the section below."
        )
    )

    news_body = research_brief(news, sentiment, ticker, company_name)
    # Reuse the researcher's classified item list under this report's heading.
    news_items = list(news_body.splitlines())
    try:
        items_start = news_items.index("## Key News Items")
        items_end = news_items.index("## Sentiment")
        news_block = "\n".join(news_items[items_start + 1 : items_end]).strip()
    except ValueError:  # pragma: no cover - defensive
        news_block = "- No classified news items available."

    kpi_lines = ["| Metric | Value | Interpretation |", "| --- | --- | --- |"]
    for row in rows:
        kpi_lines.append(f"| {row['metric']} | {row['value']} | {row['note']} |")

    chart_notes = _chart_notes(kpis)

    sentiment_block = (
        f"{sentiment_emoji(label)} **Sentiment score: {score:+.2f}** on a -1 to +1 scale "
        f"(label: {label}). Split: {sentiment.get('positive_count', 0)} positive, "
        f"{sentiment.get('neutral_count', 0)} neutral, {sentiment.get('negative_count', 0)} "
        f"negative across {sentiment.get('articles_analyzed', 0)} headlines. "
        f"The tone is driven by {_tone_driver(sentiment)}."
    )
    observations = sentiment.get("observations") or []
    if observations:
        sentiment_block += "\n\n" + "\n".join(f"- {note}" for note in observations[:3])

    risks = [
        f"Market risk: {fmt_pct(volatility, signed=False)} annualised volatility, beta "
        f"{fmt_number(kpis.get('beta'))}.",
        f"Drawdown risk: worst peak-to-trough decline of "
        f"{fmt_pct(kpis.get('max_drawdown_pct'), signed=False)} in the window.",
        f"Valuation risk: trailing P/E of {fmt_number(kpis.get('pe_ratio'))}x "
        + (
            "is elevated and vulnerable to earnings disappointment."
            if pe > 35
            else "looks broadly reasonable for the sector."
        ),
        f"News risk: aggregate sentiment is {label}; a tone shift could pressure the shares.",
    ]
    outlook_lines = [
        "\n".join(f"- {risk}" for risk in risks),
        "",
        f"**Strength:** {fmt_pct(kpis.get('period_roi_pct'))} period return, "
        f"{fmt_pct(kpis.get('momentum_20d_pct'))} 20-day momentum, trend "
        f'"{kpis.get("ma_trend")}", Sharpe {fmt_number(kpis.get("sharpe_ratio"))}.',
        f"**Watch items:** RSI {fmt_number(kpis.get('rsi_14'), 1)} "
        f"({kpis.get('rsi_signal')}), 52-week range position "
        f"{fmt_number(kpis.get('range_position_pct'), 0)}%, volume trend "
        f"{fmt_pct(kpis.get('volume_trend_pct'))}.",
        _contradiction_note(kpis, sentiment, verdict["recommendation"]),
    ]

    recommendation_lead = (
        f"the shares returned {fmt_pct(kpis.get('period_roi_pct'))} over the window, "
        f"20-day momentum is {fmt_pct(kpis.get('momentum_20d_pct'))}, the trend reads "
        f'"{kpis.get("ma_trend")}" and news sentiment is {label} '
        f"({score:+.2f}); together these give a systematic signal score of "
        f"{verdict['score']:+.2f}."
    )

    demo_banner = (
        "\n> **SYNTHETIC DEMO DATA** - figures are simulated, not market data.\n"
        if kpis.get("is_demo")
        else ""
    )
    demo_confidence_note = (
        "- Synthetic demo data caps confidence below 65.\n" if kpis.get("is_demo") else ""
    )

    report = f"""# {ticker} — Financial Analysis Report

> **Generated by the Financial Analysis Agent Crew** on {date.today().isoformat()} · analysis window {start} → {end} · data source: {kpis.get("source", "yfinance")}
{demo_banner}
## Executive Summary
{exec_summary}

## Market News Summary
{news_block}

## KPI Table and Charts
{chr(10).join(kpi_lines)}

{chr(10).join(chart_notes)}

## Sentiment Overview
{sentiment_block}

## Analysis and Outlook
{chr(10).join(outlook_lines)}

## Recommendation
**{verdict["recommendation"]}** — {recommendation_lead}

Position risk to weigh alongside this view: {fmt_pct(volatility, signed=False)} annualised
volatility, a {fmt_pct(kpis.get("max_drawdown_pct"), signed=False)} maximum drawdown and a
Sharpe ratio of {fmt_number(kpis.get("sharpe_ratio"))}.

**Confidence:** {verdict["confidence"]}/100 ({confidence_label})

{verdict["rationale"]}
{demo_confidence_note}- Missing or short history reduces confidence.
- Sentiment is headline-based and backward-looking.

## Disclaimer
{DISCLAIMER_TEXT}
"""

    if revision_note:
        report += f"\n---\n_Revision note: {revision_note}_\n"
    return report.strip()


def _chart_notes(kpis: dict[str, Any]) -> list[str]:
    """One bullet per chart, mirroring what the PDF embeds."""
    return [
        f"- **Price with moving averages** (`{kpis.get('ticker', '')}_price.png`) — SMA20 "
        f"{fmt_currency(kpis.get('sma_20'))}, SMA50 {fmt_currency(kpis.get('sma_50'))}, "
        f"SMA200 {fmt_currency(kpis.get('sma_200'))}; price above the SMA50 signals a "
        f"constructive trend.",
        "- **Volume** (`*_volume.png`) — bars coloured by up/down days; volume spikes around "
        "price moves raise the reliability of the signal.",
        "- **RSI (14)** (`*_rsi.png`) — above 70 is overbought, below 30 oversold; current "
        f"reading {fmt_number(kpis.get('rsi_14'), 1)} ({kpis.get('rsi_signal')}).",
        "- **KPI comparison** (`*_kpi.png`) — return, volatility, drawdown, RSI, P/E, Sharpe "
        "and beta on one axis for a quick risk/return read.",
    ]


def _tone_driver(sentiment: dict[str, Any]) -> str:
    counted = {
        "positive": int(sentiment.get("positive_count", 0) or 0),
        "negative": int(sentiment.get("negative_count", 0) or 0),
        "neutral": int(sentiment.get("neutral_count", 0) or 0),
    }
    dominant = max(counted, key=lambda key: counted[key]) if counted else "neutral"
    if (
        counted["positive"]
        and counted["negative"]
        and abs(counted["positive"] - counted["negative"]) <= 2
    ):
        return "a genuinely mixed set of headlines, with positive and negative items balancing"
    return f"predominantly {dominant} headlines"


def _contradiction_note(
    kpis: dict[str, Any], sentiment: dict[str, Any], recommendation: str
) -> str:
    """Call out a sentiment/price disagreement, as the Writer prompt requires."""
    score = _as_float(sentiment.get("score")) or 0.0
    momentum = _as_float(kpis.get("momentum_20d_pct")) or 0.0
    if score <= -0.25 and momentum >= 5:
        return (
            "**Divergence:** price momentum is positive while news tone is negative — "
            "treat the rally as fragile until sentiment improves."
        )
    if score >= 0.25 and momentum <= -5:
        return (
            "**Divergence:** headlines are positive but price momentum is negative — "
            "the market is not yet pricing the good news."
        )
    if recommendation in {"BUY", "STRONG BUY"} and score < 0:
        return (
            "**Divergence to monitor:** the quantitative signal is constructive while news "
            "sentiment is negative; sentiment is the swing factor."
        )
    return "**Consistency:** news tone and price trend point in the same direction."


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
