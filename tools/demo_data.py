"""Deterministic synthetic data used only when live providers are unreachable.

This module powers the app's *offline demo mode*. Every artefact produced from
it is explicitly labelled as synthetic in the UI, in the conversation log and in
the generated PDF, so a demo report can never be mistaken for market data.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from typing import Any

import numpy as np
import pandas as pd

DEMO_NOTICE = (
    "SYNTHETIC DEMO DATA — live market/news providers were unreachable or demo "
    "mode was forced. Numbers are simulated for product demonstration only and "
    "must not be used for investment decisions."
)

_SECTORS = [
    ("Technology", "Consumer Electronics"),
    ("Technology", "Software—Infrastructure"),
    ("Financial Services", "Banks—Diversified"),
    ("Healthcare", "Drug Manufacturers—General"),
    ("Consumer Cyclical", "Auto Manufacturers"),
    ("Communication Services", "Internet Content & Information"),
    ("Industrials", "Aerospace & Defense"),
    ("Energy", "Oil & Gas Integrated"),
]

_COMPANY_SUFFIXES = ["Industries", "Technologies", "Holdings", "Group", "Corporation"]


@dataclass(frozen=True)
class DemoProfile:
    """Company metadata for a synthetic ticker."""

    ticker: str
    company_name: str
    sector: str
    industry: str
    currency: str
    exchange: str


def _seed_for(ticker: str) -> int:
    return int(hashlib.sha256(ticker.encode("utf-8")).hexdigest()[:8], 16)


def demo_profile(ticker: str) -> DemoProfile:
    """Return stable, plausible company metadata for ``ticker``."""
    seed = _seed_for(ticker)
    sector, industry = _SECTORS[seed % len(_SECTORS)]
    suffix = _COMPANY_SUFFIXES[(seed >> 4) % len(_COMPANY_SUFFIXES)]
    return DemoProfile(
        ticker=ticker,
        company_name=f"{ticker} {suffix} (Demo)",
        sector=sector,
        industry=industry,
        currency="USD",
        exchange="DEMO",
    )


def demo_price_history(ticker: str, start: date, end: date) -> pd.DataFrame:
    """Generate a reproducible OHLCV frame of business days between two dates."""
    profile = demo_profile(ticker)
    seed = _seed_for(profile.ticker)
    rng = np.random.default_rng(seed)

    dates = pd.bdate_range(start=start, end=end)
    if len(dates) < 5:
        dates = pd.bdate_range(end=end, periods=5)

    base_price = 25 + (seed % 47_500) / 100  # $25 – $500
    annual_drift = ((seed >> 8) % 60 - 20) / 100  # -20% … +40% per year
    annual_vol = 0.18 + ((seed >> 16) % 30) / 100  # 18% … 48%

    steps = len(dates)
    daily_drift = annual_drift / 252
    daily_vol = annual_vol / np.sqrt(252)
    shocks = rng.normal(daily_drift, daily_vol, steps)

    # Add a gentle cyclical component so charts look realistic.
    cycle = 0.0025 * np.sin(np.linspace(0, 3 * np.pi, steps) + (seed % 7))
    close = base_price * np.exp(np.cumsum(shocks + cycle))

    open_price = close * (1 + rng.normal(0, 0.004, steps))
    high = np.maximum(open_price, close) * (1 + np.abs(rng.normal(0, 0.006, steps)))
    low = np.minimum(open_price, close) * (1 - np.abs(rng.normal(0, 0.006, steps)))
    volume = rng.integers(800_000, 25_000_000, steps)

    frame = pd.DataFrame(
        {
            "Open": open_price,
            "High": high,
            "Low": low,
            "Close": close,
            "Volume": volume.astype("int64"),
        },
        index=pd.DatetimeIndex(dates, name="Date"),
    )
    return frame.round(2)


def demo_info(ticker: str, last_price: float) -> dict[str, Any]:
    """Return synthetic company/valuation metadata for ``ticker``."""
    profile = demo_profile(ticker)
    seed = _seed_for(ticker)
    shares_outstanding = 50_000_000 + (seed % 4_000) * 1_000_000

    trailing_pe = 9 + (seed % 3_500) / 100
    eps = last_price / trailing_pe
    dividend_yield = 0.0 if seed % 3 else 0.8 + (seed % 250) / 100

    return {
        "symbol": profile.ticker,
        "longName": profile.company_name,
        "shortName": profile.ticker,
        "sector": profile.sector,
        "industry": profile.industry,
        "currency": profile.currency,
        "exchange": profile.exchange,
        "marketCap": int(shares_outstanding * last_price),
        "trailingPE": round(trailing_pe, 2),
        "forwardPE": round(trailing_pe * (0.8 + (seed % 20) / 100), 2),
        "trailingEps": round(eps, 2),
        "dividendYield": round(dividend_yield, 2),
        "beta": round(0.6 + (seed % 120) / 100, 2),
        "fiftyTwoWeekHigh": round(last_price * 1.18, 2),
        "fiftyTwoWeekLow": round(last_price * 0.72, 2),
        "averageVolume": int(1_000_000 + (seed % 20_000_000)),
        "isDemo": True,
    }


_DEMO_HEADLINES = [
    (
        "{name} beats quarterly expectations on strong demand",
        "Analysts highlighted revenue growth and improving margins after the latest earnings release.",
    ),
    (
        "{name} announces expansion plan and new partnership",
        "Management outlined a multi-year investment programme expected to broaden its product footprint.",
    ),
    (
        "Analysts adjust price targets for {name}",
        "Sell-side coverage remains mixed, with several brokers citing valuation and macro uncertainty.",
    ),
    (
        "Regulatory scrutiny intensifies for the {sector} sector",
        "Regulators opened a review that could affect compliance costs across the industry.",
    ),
    (
        "{name} shares volatile as markets digest rate outlook",
        "Trading volumes rose as investors rotated between growth and defensive names.",
    ),
    (
        "{name} insider activity draws investor attention",
        "Recent filings showed modest changes in executive holdings, with no major red flags.",
    ),
]


def demo_news(
    ticker: str, company_name: str | None = None, max_results: int = 6
) -> list[dict[str, Any]]:
    """Return synthetic news items with stable ordering for a given ticker."""
    profile = demo_profile(ticker)
    name = (company_name or profile.company_name).replace(" (Demo)", "")
    seed = _seed_for(ticker)
    now = datetime.utcnow()
    count = max(1, min(max_results, len(_DEMO_HEADLINES)))

    items: list[dict[str, Any]] = []
    for index in range(count):
        title_template, summary_template = _DEMO_HEADLINES[(seed + index) % len(_DEMO_HEADLINES)]
        percent_move = 1.2 + ((seed >> (index + 1)) % 40) / 10
        change_direction = "rose" if (seed + index) % 2 == 0 else "fell"
        items.append(
            {
                "title": title_template.format(name=name, sector=profile.sector),
                "url": "",
                "source": "Synthetic demo news",
                "published": (now - timedelta(days=index)).strftime("%Y-%m-%d"),
                "summary": (
                    f"{summary_template} In this synthetic scenario the stock "
                    f"{change_direction} {percent_move:.1f}%."
                ),
                "query": f"{ticker} demo news",
                "is_demo": True,
            }
        )
    return items


__all__ = [
    "DEMO_NOTICE",
    "DemoProfile",
    "demo_info",
    "demo_news",
    "demo_price_history",
    "demo_profile",
]
