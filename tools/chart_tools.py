"""Interactive Plotly charts plus static PNG rendering for PDF embedding.

Design notes
------------
* :func:`make_charts` builds four Plotly figures: price with moving averages,
  volume, RSI, and a KPI bar chart.
* Static export prefers Plotly's kaleido engine (pixel-identical to the web UI).
  Kaleido needs a Chrome build, which many servers lack, so we fall back to a
  ``matplotlib`` renderer that draws the same four panels. The fallback is what
  makes PDF generation work in a slim Docker image.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd

from config.settings import get_settings
from utils.logger import get_logger

logger = get_logger(__name__)

try:  # matplotlib is optional at import time; only needed for PDF fallback
    import matplotlib

    matplotlib.use("Agg")  # headless backend, safe in Streamlit/Docker
    import matplotlib.pyplot as plt
    from matplotlib.ticker import FuncFormatter

    MATPLOTLIB_AVAILABLE = True
except Exception as exc:  # pragma: no cover
    logger.warning("matplotlib unavailable (%s): PDF chart export disabled", exc)
    MATPLOTLIB_AVAILABLE = False

try:
    import plotly.graph_objects as go

    PLOTLY_AVAILABLE = True
except Exception as exc:  # pragma: no cover
    logger.warning("plotly unavailable (%s): interactive charts disabled", exc)
    PLOTLY_AVAILABLE = False

PALETTE = {
    "price": "#1f77b4",
    "sma20": "#ff7f0e",
    "sma50": "#9467bd",
    "sma200": "#7f7f7f",
    "volume": "#B0BEC5",
    "up": "#26A69A",
    "down": "#EF5350",
    "accent": "#0B6E4F",
    "grid": "#E5E7EB",
    "text": "#1F2937",
}

CHART_TITLES = {
    "price": "Price history with moving averages",
    "volume": "Trading volume",
    "rsi": "Relative Strength Index (14)",
    "kpi": "Key performance indicators",
}


@dataclass
class ChartBundle:
    """Charts produced for one analysis run."""

    figures: dict[str, Any] = field(default_factory=dict)
    images: dict[str, str] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)

    @property
    def has_images(self) -> bool:
        return bool(self.images)

    def headline_stats(self) -> dict[str, Any]:
        return {"figures": list(self.figures), "images": list(self.images)}


class ChartError(RuntimeError):
    """Raised when no chart could be produced."""


# ---------------------------------------------------------------------------
# Plotly builders
# ---------------------------------------------------------------------------
def _base_layout(fig: Any, title: str, height: int) -> Any:
    fig.update_layout(
        title={"text": title, "x": 0.02, "xanchor": "left", "font": {"size": 18}},
        template="plotly_white",
        height=height,
        margin={"l": 60, "r": 30, "t": 60, "b": 50},
        hovermode="x unified",
        font={"family": "Inter, Arial, sans-serif", "color": PALETTE["text"]},
        showlegend=True,
        legend={"orientation": "h", "y": 1.02, "x": 0.5, "xanchor": "center"},
    )
    fig.update_xaxes(showgrid=True, gridcolor=PALETTE["grid"])
    fig.update_yaxes(showgrid=True, gridcolor=PALETTE["grid"])
    return fig


def build_price_chart(
    history: pd.DataFrame, ticker: str, currency: str = "USD", height: int = 550
) -> Any:
    """Price line chart with SMA20 / SMA50 / SMA200 overlays."""
    close = history["Close"].astype(float)
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=close.index,
            y=close.values,
            name="Close",
            mode="lines",
            line={"color": PALETTE["price"], "width": 2.4},
            fill="tozeroy",
            fillcolor="rgba(31,119,180,0.08)",
        )
    )

    for window, colour, label in (
        (20, PALETTE["sma20"], "SMA 20"),
        (50, PALETTE["sma50"], "SMA 50"),
        (200, PALETTE["sma200"], "SMA 200"),
    ):
        if len(close) >= window:
            sma = close.rolling(window).mean()
            fig.add_trace(
                go.Scatter(
                    x=sma.index,
                    y=sma.values,
                    name=label,
                    mode="lines",
                    line={"color": colour, "width": 1.4, "dash": "dot"},
                )
            )

    _base_layout(fig, f"{ticker} — {CHART_TITLES['price']}", height)
    fig.update_yaxes(title_text=f"Price ({currency})", tickprefix="" if currency != "USD" else "$")
    return fig


def build_volume_chart(history: pd.DataFrame, ticker: str, height: int = 320) -> Any:
    """Volume bars coloured by up/down days."""
    if "Volume" not in history.columns:
        raise ChartError("Volume data unavailable")
    volume = history["Volume"].astype(float)
    close = history["Close"].astype(float)
    colours = [
        PALETTE["up"] if close.iloc[i] >= close.iloc[i - 1] else PALETTE["down"]
        for i in range(len(close))
    ]
    colours[0] = PALETTE["volume"]

    fig = go.Figure(go.Bar(x=volume.index, y=volume.values, marker_color=colours, name="Volume"))
    _base_layout(fig, f"{ticker} — {CHART_TITLES['volume']}", height)
    fig.update_layout(showlegend=False, bargap=0.15)
    fig.update_yaxes(title_text="Shares")
    return fig


def build_rsi_chart(history: pd.DataFrame, ticker: str, period: int = 14, height: int = 320) -> Any:
    """RSI line with 30/70 guide bands."""
    close = history["Close"].astype(float)
    delta = close.diff()
    gain = delta.clip(lower=0).ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    loss = (-delta.clip(upper=0)).ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    rsi = 100 - (100 / (1 + gain / loss.replace(0, 1e-9)))

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=rsi.index,
            y=rsi.values,
            name=f"RSI {period}",
            mode="lines",
            line={"color": PALETTE["accent"], "width": 2},
        )
    )
    fig.add_hrect(y0=70, y1=100, fillcolor="rgba(239,83,80,0.10)", line_width=0)
    fig.add_hrect(y0=0, y1=30, fillcolor="rgba(38,166,154,0.10)", line_width=0)
    fig.add_hline(y=70, line_dash="dash", line_color=PALETTE["down"], line_width=1)
    fig.add_hline(y=30, line_dash="dash", line_color=PALETTE["up"], line_width=1)

    _base_layout(fig, f"{ticker} — {CHART_TITLES['rsi']}", height)
    fig.update_layout(showlegend=False)
    fig.update_yaxes(title_text="RSI", range=[0, 100])
    return fig


def build_kpi_chart(kpis: dict[str, Any], ticker: str, height: int = 420) -> Any:
    """Bar chart of normalised KPI values (return, volatility, RSI, P/E, …)."""
    candidates: list[tuple[str, float | None, str]] = [
        ("Period ROI %", kpis.get("period_roi_pct"), "%"),
        ("Volatility %", kpis.get("volatility_annual_pct"), "%"),
        ("Max drawdown %", kpis.get("max_drawdown_pct"), "%"),
        ("RSI (14)", kpis.get("rsi_14"), ""),
        ("P/E", kpis.get("pe_ratio"), "x"),
        ("Sharpe", kpis.get("sharpe_ratio"), ""),
        ("Beta", kpis.get("beta"), ""),
    ]
    labels, values, colours, texts = [], [], [], []
    for label, raw, unit in candidates:
        try:
            number = float(raw)  # type: ignore[arg-type]
        except (TypeError, ValueError):
            continue
        if number != number:  # NaN
            continue
        labels.append(label)
        values.append(number)
        texts.append(f"{number:,.2f}{unit}")
        colours.append(PALETTE["up"] if number >= 0 else PALETTE["down"])

    if not labels:
        raise ChartError("Not enough KPI values to plot")

    fig = go.Figure(
        go.Bar(
            x=labels,
            y=values,
            text=texts,
            textposition="outside",
            marker_color=colours,
            name="KPI",
        )
    )
    _base_layout(fig, f"{ticker} — {CHART_TITLES['kpi']}", height)
    fig.update_layout(showlegend=False)
    fig.update_xaxes(tickangle=-20)
    fig.update_yaxes(title_text="Value (mixed units)")
    return fig


def make_charts(
    history: pd.DataFrame, kpis: dict[str, Any], ticker: str, currency: str = "USD"
) -> tuple[dict[str, Any], list[str]]:
    """Build every available chart, skipping (not failing) on errors.

    Returns:
        ``(figures, warnings)`` where ``figures`` maps a short key to a Plotly figure.
    """
    if not PLOTLY_AVAILABLE:
        return {}, ["Plotly is not installed; charts were skipped."]
    if history is None or history.empty:
        return {}, ["No price history available; charts were skipped."]

    settings = get_settings()
    builders: dict[str, Callable[[], Any]] = {
        "price": lambda: build_price_chart(history, ticker, currency, settings.chart_height),
        "volume": lambda: build_volume_chart(history, ticker, max(300, settings.chart_height // 2)),
        "rsi": lambda: build_rsi_chart(
            history, ticker, height=max(300, settings.chart_height // 2)
        ),
        "kpi": lambda: build_kpi_chart(kpis, ticker),
    }

    figures: dict[str, Any] = {}
    warnings: list[str] = []
    for key, builder in builders.items():
        try:
            figures[key] = builder()
        except ChartError as exc:
            warnings.append(f"{key} chart skipped: {exc}")
        except Exception as exc:  # noqa: BLE001
            logger.exception("Failed to build %s chart", key)
            warnings.append(f"{key} chart failed: {type(exc).__name__}: {exc}")
    return figures, warnings


# ---------------------------------------------------------------------------
# Static image export
# ---------------------------------------------------------------------------
def _export_with_kaleido(fig: Any, path: Path, width: int, height: int, scale: float) -> bool:
    """Try Plotly's own static export. Returns True on success."""
    try:
        fig.write_image(str(path), width=width, height=height, scale=scale, format="png")
        return path.exists() and path.stat().st_size > 1000
    except Exception as exc:  # noqa: BLE001 - kaleido raises RuntimeError without Chrome
        logger.info("Kaleido export unavailable (%s); using matplotlib fallback", exc)
        return False


def _style_matplotlib_axes(ax: Any) -> None:
    ax.grid(True, color=PALETTE["grid"], linewidth=0.6, alpha=0.9)
    ax.set_axisbelow(True)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    for spine in ("left", "bottom"):
        ax.spines[spine].set_color("#9CA3AF")


def _render_price(ax: Any, history: pd.DataFrame, ticker: str, currency: str) -> None:
    close = history["Close"].astype(float)
    ax.plot(close.index, close.values, color=PALETTE["price"], linewidth=1.8, label="Close")
    ax.fill_between(
        close.index, close.values, close.values.min(), color=PALETTE["price"], alpha=0.08
    )
    for window, colour, label in (
        (20, PALETTE["sma20"], "SMA 20"),
        (50, PALETTE["sma50"], "SMA 50"),
        (200, PALETTE["sma200"], "SMA 200"),
    ):
        if len(close) >= window:
            ax.plot(
                close.index,
                close.rolling(window).mean().values,
                color=colour,
                linewidth=1.1,
                linestyle=":",
                label=label,
            )
    ax.set_title(f"{ticker} — price with moving averages", fontsize=13, loc="left", weight="bold")
    ax.set_ylabel(f"Price ({currency})")
    ax.legend(loc="upper left", fontsize=8, frameon=False, ncol=2)
    _style_matplotlib_axes(ax)


def _render_volume(ax: Any, history: pd.DataFrame, ticker: str) -> None:
    volume = history["Volume"].astype(float)
    close = history["Close"].astype(float)
    colours = [
        PALETTE["up"] if close.iloc[i] >= close.iloc[i - 1] else PALETTE["down"]
        for i in range(len(close))
    ]
    ax.bar(volume.index, volume.values, color=colours, width=0.8)
    ax.set_title(f"{ticker} — trading volume", fontsize=12, loc="left", weight="bold")
    ax.yaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value / 1e6:,.0f}M"))
    _style_matplotlib_axes(ax)


def _render_rsi(ax: Any, history: pd.DataFrame, ticker: str, period: int = 14) -> None:
    close = history["Close"].astype(float)
    delta = close.diff()
    gain = delta.clip(lower=0).ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    loss = (-delta.clip(upper=0)).ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    rsi = 100 - (100 / (1 + gain / loss.replace(0, 1e-9)))
    ax.plot(rsi.index, rsi.values, color=PALETTE["accent"], linewidth=1.6)
    ax.axhspan(70, 100, color=PALETTE["down"], alpha=0.12)
    ax.axhspan(0, 30, color=PALETTE["up"], alpha=0.12)
    ax.axhline(70, color=PALETTE["down"], linestyle="--", linewidth=0.8)
    ax.axhline(30, color=PALETTE["up"], linestyle="--", linewidth=0.8)
    ax.set_ylim(0, 100)
    ax.set_title(f"{ticker} — RSI ({period})", fontsize=12, loc="left", weight="bold")
    _style_matplotlib_axes(ax)


def _render_kpi(ax: Any, kpis: dict[str, Any], ticker: str) -> None:
    candidates = [
        ("ROI %", kpis.get("period_roi_pct")),
        ("Vol %", kpis.get("volatility_annual_pct")),
        ("Drawdown %", kpis.get("max_drawdown_pct")),
        ("RSI", kpis.get("rsi_14")),
        ("P/E", kpis.get("pe_ratio")),
        ("Sharpe", kpis.get("sharpe_ratio")),
        ("Beta", kpis.get("beta")),
    ]
    labels, values = [], []
    for label, raw in candidates:
        try:
            number = float(raw)  # type: ignore[arg-type]
        except (TypeError, ValueError):
            continue
        if number != number:
            continue
        labels.append(label)
        values.append(number)

    if not labels:
        raise ChartError("Not enough KPI values to plot")

    colours = [PALETTE["up"] if value >= 0 else PALETTE["down"] for value in values]
    bars = ax.bar(labels, values, color=colours)
    for bar, value in zip(bars, values, strict=True):
        ax.annotate(
            f"{value:,.1f}",
            (bar.get_x() + bar.get_width() / 2, value),
            ha="center",
            va="bottom" if value >= 0 else "top",
            fontsize=8,
        )
    ax.axhline(0, color="#6B7280", linewidth=0.8)
    ax.set_title(f"{ticker} — key performance indicators", fontsize=12, loc="left", weight="bold")
    ax.tick_params(axis="x", labelsize=8)
    _style_matplotlib_axes(ax)


_RENDERERS: dict[str, Callable[..., None]] = {
    "price": _render_price,
    "volume": _render_volume,
    "rsi": _render_rsi,
    "kpi": _render_kpi,
}


def _export_with_matplotlib(
    key: str,
    history: pd.DataFrame,
    kpis: dict[str, Any],
    ticker: str,
    currency: str,
    path: Path,
) -> bool:
    """Render a chart with matplotlib and save a PNG. Returns True on success."""
    if not MATPLOTLIB_AVAILABLE:
        return False
    renderer = _RENDERERS.get(key)
    if renderer is None:
        return False

    fig, ax = plt.subplots(figsize=(11, 5.0 if key == "price" else 3.2), dpi=150)
    try:
        if key == "kpi":
            renderer(ax, kpis, ticker)
        elif key == "price":
            renderer(ax, history, ticker, currency)
        else:
            renderer(ax, history, ticker)
        fig.autofmt_xdate(rotation=20)
        fig.tight_layout()
        fig.savefig(path, format="png", bbox_inches="tight", facecolor="white")
        return path.exists() and path.stat().st_size > 1000
    except Exception as exc:  # noqa: BLE001
        logger.warning("matplotlib export failed for %s chart: %s", key, exc)
        return False
    finally:
        plt.close(fig)


def save_chart_images(
    figures: dict[str, Any],
    history: pd.DataFrame,
    kpis: dict[str, Any],
    ticker: str,
    currency: str = "USD",
    output_dir: str | Path | None = None,
    run_id: str | None = None,
) -> tuple[dict[str, str], list[str]]:
    """Persist charts as PNG files suitable for embedding in the PDF.

    Args:
        figures: Plotly figures keyed by chart name (``price``, ``volume``, …).
        history: Price history, used by the matplotlib fallback renderer.
        kpis: KPI dict, used by the KPI chart renderer.
        ticker: Ticker symbol, used in filenames and titles.
        currency: Currency code shown on the price axis.
        output_dir: Destination directory (defaults to the configured chart dir).
        run_id: Optional suffix to keep runs separate.

    Returns:
        ``(image_paths, warnings)`` where ``image_paths`` maps chart key → file path.
    """
    settings = get_settings()
    destination = Path(output_dir) if output_dir else settings.chart_dir
    destination.mkdir(parents=True, exist_ok=True)
    stamp = run_id or datetime.now().strftime("%Y%m%d-%H%M%S")
    safe_ticker = "".join(char for char in ticker.upper() if char.isalnum() or char in ".-")

    images: dict[str, str] = {}
    warnings: list[str] = []

    for key in ("price", "volume", "rsi", "kpi"):
        path = destination / f"{safe_ticker}_{key}_{stamp}.png"
        figure = figures.get(key)
        if figure is not None and _export_with_kaleido(
            figure, path, settings.chart_width, settings.chart_height, settings.chart_scale
        ):
            images[key] = str(path)
            continue

        if _export_with_matplotlib(key, history, kpis, ticker, currency, path):
            images[key] = str(path)
        elif figure is not None or key in _RENDERERS:
            warnings.append(f"Could not render the {key} chart as an image.")

    if not images:
        warnings.append("No chart images could be rendered for the PDF report.")
    return images, warnings


def chart_to_html(figure: Any, include_plotlyjs: str | bool = "cdn") -> str:
    """Return the embeddable HTML for a Plotly figure (used by exports)."""
    return figure.to_html(include_plotlyjs=include_plotlyjs, full_html=False)


def get_market_chart(ticker: str, start_date: str, end_date: str) -> str:
    """Build interactive price and KPI charts for a ticker and save them as PNGs.

    Use this tool when the final report needs visual evidence.

    Args:
        ticker: Stock ticker symbol, e.g. "AAPL".
        start_date: Start date in YYYY-MM-DD format.
        end_date: End date in YYYY-MM-DD format.

    Returns:
        A JSON string describing the generated charts and their saved locations.
    """
    from datetime import datetime as _dt

    from tools.finance_tools import compute_kpis, fetch_stock_data

    try:
        start = _dt.strptime(start_date, "%Y-%m-%d").date()
        end = _dt.strptime(end_date, "%Y-%m-%d").date()
        stock = fetch_stock_data(ticker, start, end)
        kpis = compute_kpis(stock)
        figures, warnings = make_charts(stock.history, kpis, stock.ticker, stock.currency)
        images, image_warnings = save_chart_images(
            figures, stock.history, kpis, stock.ticker, stock.currency
        )
        return json.dumps(
            {
                "ticker": stock.ticker,
                "charts": list(figures),
                "images": images,
                "warnings": warnings + image_warnings,
                "is_demo": stock.is_demo,
            },
            default=str,
        )
    except Exception as exc:  # noqa: BLE001
        logger.exception("get_market_chart failed for %s", ticker)
        return json.dumps({"error": f"{type(exc).__name__}: {exc}", "ticker": ticker})


__all__ = [
    "CHART_TITLES",
    "ChartBundle",
    "ChartError",
    "build_kpi_chart",
    "build_price_chart",
    "build_rsi_chart",
    "build_volume_chart",
    "chart_to_html",
    "get_market_chart",
    "make_charts",
    "save_chart_images",
]
