"""Market-data retrieval and quantitative KPI computation (yfinance).

The public surface is intentionally split in two:

* Plain Python functions (:func:`fetch_stock_data`, :func:`compute_kpis`, …)
  used by the orchestrator and the tests. They return dataclasses / dicts.
* Thin JSON-returning wrappers (:func:`get_stock_snapshot`,
  :func:`get_financial_kpis`) that are registered as tools on the agno agents.

Live data can fail (no network, delisted ticker, rate limits). When that happens
the module either raises :class:`FinanceDataError` or, if demo fallback is
enabled, returns clearly labelled synthetic data.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
from typing import Any

import pandas as pd

from config.settings import get_settings
from tools.demo_data import DEMO_NOTICE, demo_info, demo_price_history, demo_profile
from utils.formatting import safe_float
from utils.logger import get_logger

logger = get_logger(__name__)

DEFAULT_BENCHMARK = "^GSPC"
MIN_HISTORY_ROWS = 5


class FinanceDataError(RuntimeError):
    """Raised when market data cannot be retrieved or is unusable."""


@dataclass
class StockData:
    """Price history plus company metadata for one ticker."""

    ticker: str
    company_name: str
    start: date
    end: date
    history: pd.DataFrame
    info: dict[str, Any] = field(default_factory=dict)
    currency: str = "USD"
    exchange: str = ""
    sector: str = "Unknown"
    industry: str = "Unknown"
    is_demo: bool = False
    source: str = "yfinance"
    warnings: list[str] = field(default_factory=list)

    @property
    def has_data(self) -> bool:
        return not self.history.empty and "Close" in self.history.columns

    @property
    def latest_price(self) -> float | None:
        if not self.has_data:
            return None
        return safe_float(self.history["Close"].dropna().iloc[-1])


# ---------------------------------------------------------------------------
# Data retrieval
# ---------------------------------------------------------------------------
def fetch_stock_data(
    ticker: str,
    start: date,
    end: date,
    *,
    allow_demo: bool | None = None,
    force_demo: bool | None = None,
) -> StockData:
    """Fetch daily OHLCV history and company metadata for ``ticker``.

    Args:
        ticker: Upper-case ticker symbol, e.g. ``AAPL``.
        start: Inclusive start date.
        end: Inclusive end date (yfinance is exclusive, so one day is added).
        allow_demo: Override settings for synthetic fallback.
        force_demo: Skip the network entirely and return synthetic data.

    Returns:
        A :class:`StockData` instance.

    Raises:
        FinanceDataError: If live data is unavailable and demo fallback is off.
    """
    settings = get_settings()
    allow_demo = settings.allow_demo_data if allow_demo is None else allow_demo
    force_demo = settings.force_demo_data if force_demo is None else force_demo
    ticker = ticker.upper().strip()

    if force_demo:
        logger.info("Demo mode forced for %s — skipping live market data", ticker)
        return _build_demo_stock(ticker, start, end, "demo mode forced (FORCE_DEMO_DATA=1)")

    try:
        stock = _fetch_yfinance(ticker, start, end)
        # A window of only holidays/weekends can legitimately come back empty.
        if stock.has_data:
            logger.info(
                "Fetched %s rows of %s data for %s (%s → %s)",
                len(stock.history),
                stock.source,
                ticker,
                start,
                end,
            )
            return stock
        reason = "provider returned an empty price history"
    except Exception as exc:  # noqa: BLE001 - third-party library raises broadly
        reason = f"{type(exc).__name__}: {exc}"
        logger.warning("Live market data failed for %s: %s", ticker, reason)

    if allow_demo:
        logger.warning("Falling back to synthetic demo data for %s (%s)", ticker, reason)
        return _build_demo_stock(ticker, start, end, reason)

    raise FinanceDataError(
        f"Could not retrieve market data for '{ticker}' ({reason}). "
        "Check the ticker, your internet connection, or enable ALLOW_DEMO_DATA=true "
        "for an offline demo."
    )


def _fetch_yfinance(ticker: str, start: date, end: date) -> StockData:
    """Internal live fetch using yfinance; raises on any provider error."""
    import yfinance as yf  # imported lazily to keep CLI/test startup fast

    settings = get_settings()
    ticker_obj = yf.Ticker(ticker)

    # yfinance treats ``end`` as exclusive.
    history = ticker_obj.history(
        start=start.isoformat(),
        end=(end + timedelta(days=1)).isoformat(),
        interval="1d",
        auto_adjust=True,
        actions=False,
        timeout=settings.request_timeout_seconds,
    )

    if history is None or history.empty:
        raise FinanceDataError("no rows returned")

    history = history.copy()
    history.index = pd.to_datetime(history.index).tz_localize(None)
    history.index.name = "Date"
    keep = [col for col in ("Open", "High", "Low", "Close", "Volume") if col in history.columns]
    history = history[keep].dropna(subset=["Close"]).sort_index()

    info = _safe_info(ticker_obj)
    profile = _safe_profile(ticker_obj, info)

    return StockData(
        ticker=ticker,
        company_name=profile["company_name"],
        start=start,
        end=end,
        history=history,
        info=info,
        currency=profile["currency"],
        exchange=profile["exchange"],
        sector=profile["sector"],
        industry=profile["industry"],
        is_demo=False,
        source="yfinance",
    )


def _safe_info(ticker_obj: Any) -> dict[str, Any]:
    """Fetch ``Ticker.info`` without letting metadata failures break a run."""
    for attribute in ("info", "fast_info"):
        try:
            value = getattr(ticker_obj, attribute)
            if attribute == "fast_info":
                value = dict(value) if value else {}
            if value:
                return dict(value)
        except Exception as exc:  # noqa: BLE001
            logger.debug("yfinance %s unavailable: %s", attribute, exc)
    logger.warning("Company metadata unavailable; continuing with price data only")
    return {}


def _safe_profile(ticker_obj: Any, info: dict[str, Any]) -> dict[str, str]:
    """Best-effort company profile lookup with sensible fallbacks."""
    company_name = (
        info.get("longName")
        or info.get("shortName")
        or info.get("displayName")
        or info.get("symbol")
        or ""
    )
    currency = str(info.get("currency") or info.get("currencyCode") or "USD")
    exchange = str(info.get("exchange") or info.get("exchangeName") or "")
    sector = str(info.get("sector") or "Unknown")
    industry = str(info.get("industry") or "Unknown")

    if not company_name:
        try:
            profile = ticker_obj.get_profile()  # newer yfinance versions
            company_name = profile.get("companyName") or profile.get("name") or ""
        except Exception:  # noqa: BLE001
            company_name = ""

    return {
        "company_name": company_name or ticker_obj.ticker,
        "currency": currency,
        "exchange": exchange,
        "sector": sector,
        "industry": industry,
    }


def _build_demo_stock(ticker: str, start: date, end: date, reason: str) -> StockData:
    """Construct a :class:`StockData` from synthetic demo data."""
    profile = demo_profile(ticker)
    history = demo_price_history(ticker, start, end)
    last_price = float(history["Close"].iloc[-1])
    return StockData(
        ticker=ticker,
        company_name=profile.company_name,
        start=start,
        end=end,
        history=history,
        info=demo_info(ticker, last_price),
        currency=profile.currency,
        exchange=profile.exchange,
        sector=profile.sector,
        industry=profile.industry,
        is_demo=True,
        source="synthetic-demo",
        warnings=[
            DEMO_NOTICE,
            f"Live data provider unreachable: {reason}",
            "Set ALLOW_DEMO_DATA=false and FORCE_DEMO_DATA=0 to disable this fallback.",
        ],
    )


def fetch_benchmark(
    benchmark_ticker: str,
    start: date,
    end: date,
    *,
    allow_demo: bool = True,
) -> pd.DataFrame | None:
    """Fetch benchmark prices for relative-performance metrics.

    Returns ``None`` instead of raising so a missing benchmark never blocks a
    report. Synthetic demo benchmarks are deliberately *not* mixed with real
    stock data.
    """
    if not benchmark_ticker:
        return None
    try:
        import yfinance as yf

        settings = get_settings()
        frame = yf.Ticker(benchmark_ticker).history(
            start=start.isoformat(),
            end=(end + timedelta(days=1)).isoformat(),
            interval="1d",
            auto_adjust=True,
            actions=False,
            timeout=settings.request_timeout_seconds,
        )
        if frame is None or frame.empty:
            return None
        frame.index = pd.to_datetime(frame.index).tz_localize(None)
        return frame[["Close"]].dropna().sort_index()
    except Exception as exc:  # noqa: BLE001
        logger.warning("Benchmark %s unavailable: %s", benchmark_ticker, exc)
        return None


# ---------------------------------------------------------------------------
# Indicator maths
# ---------------------------------------------------------------------------
def compute_rsi(close: pd.Series, period: int = 14) -> float | None:
    """Wilder's Relative Strength Index for the latest bar.

    Args:
        close: Chronologically ordered closing prices.
        period: Look-back window (14 by default).

    Returns:
        RSI in ``[0, 100]`` or ``None`` when there is not enough history.
    """
    series = pd.Series(close).dropna().astype(float)
    if len(series) < period + 1:
        return None

    delta = series.diff()
    gain = delta.clip(lower=0.0)
    loss = -delta.clip(upper=0.0)

    avg_gain = gain.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()

    latest_gain = avg_gain.iloc[-1]
    latest_loss = avg_loss.iloc[-1]
    if latest_loss == 0:
        return 100.0 if latest_gain > 0 else 50.0
    rs = latest_gain / latest_loss
    return float(round(100 - (100 / (1 + rs)), 2))


def compute_max_drawdown(close: pd.Series) -> float | None:
    """Maximum peak-to-trough drawdown as a positive percentage."""
    series = pd.Series(close).dropna().astype(float)
    if len(series) < 2:
        return None
    running_max = series.cummax()
    drawdown = (series / running_max - 1.0) * 100
    return float(round(abs(drawdown.min()), 2))


def annualized_volatility(close: pd.Series, trading_days: int = 252) -> float | None:
    """Annualised standard deviation of daily returns, in percent."""
    series = pd.Series(close).dropna().astype(float)
    if len(series) < 3:
        return None
    returns = series.pct_change().dropna()
    if returns.empty:
        return None
    return float(round(returns.std(ddof=1) * math.sqrt(trading_days) * 100, 2))


def sharpe_ratio(
    close: pd.Series, risk_free_rate: float = 0.04, trading_days: int = 252
) -> float | None:
    """Annualised Sharpe ratio using daily returns."""
    series = pd.Series(close).dropna().astype(float)
    if len(series) < 3:
        return None
    returns = series.pct_change().dropna()
    if returns.empty or returns.std(ddof=1) == 0:
        return None
    excess = returns.mean() - (risk_free_rate / trading_days)
    return float(round(excess / returns.std(ddof=1) * math.sqrt(trading_days), 2))


def _period_return_pct(close: pd.Series) -> float | None:
    series = pd.Series(close).dropna().astype(float)
    if len(series) < 2 or series.iloc[0] == 0:
        return None
    return float(round((series.iloc[-1] / series.iloc[0] - 1) * 100, 2))


def _annualized_return_pct(close: pd.Series, trading_days: int = 252) -> float | None:
    series = pd.Series(close).dropna().astype(float)
    if len(series) < 2 or series.iloc[0] <= 0:
        return None
    total = series.iloc[-1] / series.iloc[0]
    years = len(series) / trading_days
    if years <= 0:
        return None
    try:
        return float(round((total ** (1 / years) - 1) * 100, 2))
    except (OverflowError, ValueError):
        return None


def _relative_return(close: pd.Series, benchmark: pd.Series | None) -> float | None:
    if benchmark is None or benchmark.empty:
        return None
    stock_return = _period_return_pct(close)
    bench_return = _period_return_pct(benchmark)
    if stock_return is None or bench_return is None:
        return None
    return float(round(stock_return - bench_return, 2))


def _momentum_pct(close: pd.Series, window: int) -> float | None:
    series = pd.Series(close).dropna().astype(float)
    if len(series) <= window:
        return None
    past = series.iloc[-window - 1]
    if past == 0:
        return None
    return float(round((series.iloc[-1] / past - 1) * 100, 2))


def _moving_average_signal(
    price: float | None, sma20: float | None, sma50: float | None
) -> tuple[str, bool]:
    """Classify the short/medium-term trend and detect a golden cross."""
    if price is None or sma20 is None or sma50 is None:
        return "Insufficient history", False
    golden_cross = sma20 > sma50
    if price > sma20 > sma50:
        return "Bullish (price > SMA20 > SMA50)", golden_cross
    if price < sma20 < sma50:
        return "Bearish (price < SMA20 < SMA50)", golden_cross
    if price > sma50:
        return "Mildly bullish (price above SMA50)", golden_cross
    return "Mildly bearish (price below SMA50)", golden_cross


def _annualised_stdev_pct(series: pd.Series, trading_days: int) -> float | None:
    values = pd.Series(series).dropna()
    if len(values) < 3:
        return None
    return float(round(values.std(ddof=1) * math.sqrt(trading_days) * 100, 2))


# ---------------------------------------------------------------------------
# KPI computation
# ---------------------------------------------------------------------------
def compute_kpis(
    stock: StockData,
    benchmark: pd.DataFrame | None = None,
    benchmark_ticker: str = DEFAULT_BENCHMARK,
) -> dict[str, Any]:
    """Compute valuation, risk, momentum and liquidity metrics.

    Args:
        stock: Price history + metadata from :func:`fetch_stock_data`.
        benchmark: Optional benchmark close prices for relative performance.
        benchmark_ticker: Benchmark symbol (recorded in the output).

    Returns:
        A JSON-friendly dict of KPIs plus a ``data_quality`` note list.
    """
    settings = get_settings()
    if not stock.has_data:
        raise FinanceDataError(f"No usable price history for {stock.ticker}")

    history = stock.history
    close = history["Close"].astype(float)
    volume = history["Volume"].astype(float) if "Volume" in history else pd.Series(dtype=float)
    info = stock.info or {}
    trading_days = settings.trading_days

    price = safe_float(close.iloc[-1])
    start_price = safe_float(close.iloc[0])
    period_roi = _period_return_pct(close)
    annual_return = _annualized_return_pct(close, trading_days)
    volatility = annualized_volatility(close, trading_days)
    sharpe = sharpe_ratio(close, settings.risk_free_rate, trading_days)
    drawdown = compute_max_drawdown(close)
    rsi = compute_rsi(close)

    daily_returns = close.pct_change().dropna()
    up_day_ratio = (
        float(round((daily_returns > 0).mean() * 100, 2)) if not daily_returns.empty else None
    )

    sma20 = safe_float(close.rolling(20).mean().iloc[-1]) if len(close) >= 20 else None
    sma50 = safe_float(close.rolling(50).mean().iloc[-1]) if len(close) >= 50 else None
    sma200 = safe_float(close.rolling(200).mean().iloc[-1]) if len(close) >= 200 else None

    benchmark_close: pd.Series | None = None
    if benchmark is not None and not benchmark.empty:
        benchmark_close = benchmark["Close"].astype(float)

    high_52w = safe_float(info.get("fiftyTwoWeekHigh")) or safe_float(close.max())
    low_52w = safe_float(info.get("fiftyTwoWeekLow")) or safe_float(close.min())
    range_position = None
    if high_52w is not None and low_52w is not None and high_52w > low_52w and price is not None:
        range_position = float(round((price - low_52w) / (high_52w - low_52w) * 100, 2))

    trend_label, golden_cross = _moving_average_signal(price, sma20, sma50)

    volume_trend = None
    if len(volume) >= 30:
        recent = volume.tail(10).mean()
        baseline = volume.tail(60).mean() if len(volume) >= 60 else volume.mean()
        if baseline and baseline > 0:
            volume_trend = float(round((recent / baseline - 1) * 100, 2))

    data_quality: list[str] = []
    if stock.is_demo:
        data_quality.append(DEMO_NOTICE)
    if sma200 is None:
        data_quality.append("SMA200 unavailable (fewer than 200 trading days in range).")
    if info.get("trailingPE") is None:
        data_quality.append("P/E unavailable from provider (often true for loss-making companies).")
    if len(close) < 60:
        data_quality.append(
            "Short history: volatility and momentum are less reliable below 60 bars."
        )
    if benchmark_close is None:
        data_quality.append(f"Benchmark {benchmark_ticker} unavailable; relative return omitted.")
    missing_profile = info.get("sector") in (None, "", "Unknown")
    if missing_profile:
        data_quality.append("Sector/industry metadata unavailable.")

    dividend_yield = safe_float(info.get("dividendYield"))
    if dividend_yield is not None and dividend_yield < 1:  # yfinance uses fractions
        dividend_yield *= 100

    return {
        "ticker": stock.ticker,
        "company_name": stock.company_name,
        "sector": stock.sector,
        "industry": stock.industry,
        "exchange": stock.exchange,
        "currency": stock.currency,
        "source": stock.source,
        "is_demo": stock.is_demo,
        "start": stock.start.isoformat(),
        "end": stock.end.isoformat(),
        "trading_days": int(len(close)),
        "current_price": price,
        "start_price": start_price,
        "price_change_abs": (
            float(round(price - start_price, 2))
            if price is not None and start_price is not None
            else None
        ),
        "period_roi_pct": period_roi,
        "annualized_return_pct": annual_return,
        "volatility_annual_pct": volatility,
        "daily_volatility_pct": _annualised_stdev_pct(daily_returns, 1),
        "sharpe_ratio": sharpe,
        "max_drawdown_pct": drawdown,
        "rsi_14": rsi,
        "rsi_signal": _rsi_signal(rsi),
        "sma_20": sma20,
        "sma_50": sma50,
        "sma_200": sma200,
        "price_vs_sma20_pct": _pct_gap(price, sma20),
        "price_vs_sma50_pct": _pct_gap(price, sma50),
        "price_vs_sma200_pct": _pct_gap(price, sma200),
        "ma_trend": trend_label,
        "golden_cross": golden_cross,
        "momentum_20d_pct": _momentum_pct(close, 20),
        "momentum_60d_pct": _momentum_pct(close, 60),
        "up_day_ratio_pct": up_day_ratio,
        "best_day_pct": (
            float(round(daily_returns.max() * 100, 2)) if not daily_returns.empty else None
        ),
        "worst_day_pct": (
            float(round(daily_returns.min() * 100, 2)) if not daily_returns.empty else None
        ),
        "market_cap": safe_float(info.get("marketCap")),
        "pe_ratio": safe_float(info.get("trailingPE")),
        "forward_pe": safe_float(info.get("forwardPE")),
        "eps": safe_float(info.get("trailingEps")),
        "dividend_yield_pct": dividend_yield,
        "beta": safe_float(info.get("beta")),
        "avg_volume": safe_float(info.get("averageVolume"))
        or (float(round(volume.tail(30).mean(), 0)) if not volume.empty else None),
        "high_52w": high_52w,
        "low_52w": low_52w,
        "range_position_pct": range_position,
        "volume_trend_pct": volume_trend,
        "benchmark_ticker": benchmark_ticker,
        "benchmark_roi_pct": _period_return_pct(benchmark_close)
        if benchmark_close is not None
        else None,
        "relative_return_pct": _relative_return(close, benchmark_close),
        "risk_free_rate_pct": round(settings.risk_free_rate * 100, 2),
        "data_quality": data_quality,
    }


def _pct_gap(price: float | None, reference: float | None) -> float | None:
    if price is None or not reference:
        return None
    return float(round((price / reference - 1) * 100, 2))


def _rsi_signal(rsi: float | None) -> str:
    if rsi is None:
        return "n/a"
    if rsi >= 70:
        return "Overbought"
    if rsi <= 30:
        return "Oversold"
    if rsi >= 55:
        return "Bullish momentum"
    if rsi <= 45:
        return "Bearish momentum"
    return "Neutral"


def build_kpi_table(kpis: dict[str, Any]) -> list[dict[str, str]]:
    """Convert raw KPIs into human-readable ``Metric / Value / Note`` rows."""
    from utils.formatting import fmt_compact, fmt_currency, fmt_number, fmt_pct

    currency = "$" if kpis.get("currency", "USD") == "USD" else f"{kpis.get('currency')} "
    rows = [
        ("Current price", fmt_currency(kpis.get("current_price"), symbol=currency), "Latest close"),
        ("Period return (ROI)", fmt_pct(kpis.get("period_roi_pct")), "Start → end of range"),
        ("Annualised return", fmt_pct(kpis.get("annualized_return_pct")), "Compounded"),
        (
            "Benchmark return",
            fmt_pct(kpis.get("benchmark_roi_pct")),
            str(kpis.get("benchmark_ticker") or "—"),
        ),
        ("Relative return", fmt_pct(kpis.get("relative_return_pct")), "Stock − benchmark"),
        (
            "Volatility (annual)",
            fmt_pct(kpis.get("volatility_annual_pct"), signed=False),
            "Stdev of daily returns × √252",
        ),
        (
            "Sharpe ratio",
            fmt_number(kpis.get("sharpe_ratio")),
            f"Rf = {kpis.get('risk_free_rate_pct')}%",
        ),
        ("Max drawdown", fmt_pct(kpis.get("max_drawdown_pct"), signed=False), "Peak-to-trough"),
        ("RSI (14)", fmt_number(kpis.get("rsi_14"), 1), str(kpis.get("rsi_signal") or "")),
        ("SMA 20", fmt_currency(kpis.get("sma_20"), symbol=currency), "Short-term trend"),
        ("SMA 50", fmt_currency(kpis.get("sma_50"), symbol=currency), "Medium-term trend"),
        ("SMA 200", fmt_currency(kpis.get("sma_200"), symbol=currency), "Long-term trend"),
        ("Trend signal", str(kpis.get("ma_trend") or "n/a"), "Moving-average structure"),
        ("P/E (trailing)", fmt_number(kpis.get("pe_ratio")), "Valuation"),
        ("Forward P/E", fmt_number(kpis.get("forward_pe")), "Forward estimate"),
        ("EPS (trailing)", fmt_currency(kpis.get("eps"), symbol=currency), "Earnings per share"),
        ("Dividend yield", fmt_pct(kpis.get("dividend_yield_pct"), signed=False), "Annual"),
        ("Beta", fmt_number(kpis.get("beta")), "Market sensitivity"),
        ("Market cap", fmt_compact(kpis.get("market_cap"), symbol=currency), "Company size"),
        ("Avg volume", fmt_compact(kpis.get("avg_volume"), symbol=""), "Liquidity"),
        (
            "52-week range",
            f"{fmt_currency(kpis.get('low_52w'), symbol=currency)} – {fmt_currency(kpis.get('high_52w'), symbol=currency)}",
            "Low / high",
        ),
        ("20-day momentum", fmt_pct(kpis.get("momentum_20d_pct")), "Short-term"),
        ("60-day momentum", fmt_pct(kpis.get("momentum_60d_pct")), "Medium-term"),
    ]
    return [{"metric": metric, "value": value, "note": note} for metric, value, note in rows]


# ---------------------------------------------------------------------------
# agno tool wrappers (must return JSON-serialisable strings)
# ---------------------------------------------------------------------------
def get_stock_snapshot(ticker: str, start_date: str, end_date: str) -> str:
    """Fetch recent price history and company metadata for a stock ticker.

    Use this tool to ground your analysis in real market data.

    Args:
        ticker: Stock ticker symbol, e.g. "AAPL" or "TSLA".
        start_date: Start date in YYYY-MM-DD format.
        end_date: End date in YYYY-MM-DD format.

    Returns:
        A JSON string with company profile, latest price, and recent closes.
    """
    try:
        start = datetime.strptime(start_date, "%Y-%m-%d").date()
        end = datetime.strptime(end_date, "%Y-%m-%d").date()
        stock = fetch_stock_data(ticker, start, end)
        recent = stock.history.tail(15).reset_index()
        recent["Date"] = recent["Date"].dt.strftime("%Y-%m-%d")
        payload = {
            "ticker": stock.ticker,
            "company_name": stock.company_name,
            "sector": stock.sector,
            "industry": stock.industry,
            "currency": stock.currency,
            "exchange": stock.exchange,
            "source": stock.source,
            "is_demo": stock.is_demo,
            "latest_price": stock.latest_price,
            "rows": int(len(stock.history)),
            "recent_closes": recent.round(2).to_dict(orient="records"),
            "warnings": stock.warnings,
        }
        return json.dumps(payload, default=str)
    except Exception as exc:  # noqa: BLE001
        logger.exception("get_stock_snapshot failed for %s", ticker)
        return json.dumps({"error": f"{type(exc).__name__}: {exc}", "ticker": ticker})


def get_financial_kpis(ticker: str, start_date: str, end_date: str) -> str:
    """Compute quantitative KPIs (valuation, risk, momentum) for a stock.

    Args:
        ticker: Stock ticker symbol, e.g. "AAPL".
        start_date: Start date in YYYY-MM-DD format.
        end_date: End date in YYYY-MM-DD format.

    Returns:
        A JSON string with P/E, ROI, volatility, RSI, moving averages and more.
    """
    try:
        start = datetime.strptime(start_date, "%Y-%m-%d").date()
        end = datetime.strptime(end_date, "%Y-%m-%d").date()
        stock = fetch_stock_data(ticker, start, end)
        benchmark = None
        if not stock.is_demo:
            benchmark = fetch_benchmark(DEFAULT_BENCHMARK, start, end)
        kpis = compute_kpis(stock, benchmark)
        return json.dumps(kpis, default=str)
    except Exception as exc:  # noqa: BLE001
        logger.exception("get_financial_kpis failed for %s", ticker)
        return json.dumps({"error": f"{type(exc).__name__}: {exc}", "ticker": ticker})


__all__ = [
    "DEFAULT_BENCHMARK",
    "FinanceDataError",
    "StockData",
    "annualized_volatility",
    "build_kpi_table",
    "compute_kpis",
    "compute_max_drawdown",
    "compute_rsi",
    "fetch_benchmark",
    "fetch_stock_data",
    "get_financial_kpis",
    "get_stock_snapshot",
    "sharpe_ratio",
]
