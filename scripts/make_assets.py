#!/usr/bin/env python3
"""Generate the README's visual assets from the project's own output.

Everything here is reproducible and offline (no API keys, no network, no browser):

* ``docs/assets/banner.svg``            - hero banner (crisp SVG, subtle SMIL shimmer)
* ``docs/assets/architecture.svg``      - the crew's pipeline diagram
* ``docs/assets/demo.gif``              - animated run: agents fire in sequence, charts draw,
                                          a report is stamped "PDF ready"
* ``docs/assets/report-preview.png``    - page previews rendered from a **real** generated PDF
* ``docs/assets/charts-preview.png``    - the four **real** chart PNGs composed into a grid

Usage::

    python scripts/make_assets.py            # writes docs/assets/*
    python scripts/make_assets.py --keep-pdf # keep the temporary PDF in reports/output

The animated GIF and the previews use deterministic synthetic (demo) data, so the
images are byte-stable across runs apart from timestamps embedded in the PDF.
"""

from __future__ import annotations

import argparse
import os
import sys
from dataclasses import dataclass
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

os.environ.setdefault("FORCE_DEMO_DATA", "1")
os.environ.setdefault("LOG_LEVEL", "WARNING")

ASSETS = ROOT / "docs" / "assets"

# Brand palette (kept in sync with app.py / the PDF generator).
NAVY = "#0B3D91"
NAVY_DARK = "#081FB0" if False else "#071C4A"
GREEN = "#0B6E4F"
TEAL = "#26A69A"
RED = "#C0392B"
INK = "#0B1220"
INK_SOFT = "#111C33"
MUTED = "#8FA3BF"
WHITE = "#FFFFFF"
AMBER = "#E8B54D"


# ---------------------------------------------------------------------------
# Static images (matplotlib + Pillow)
# ---------------------------------------------------------------------------
def _rounded_alpha(image, radius: int = 22):
    """Give a rendered image rounded corners with a transparent alpha channel."""
    from PIL import Image, ImageDraw

    mask = Image.new("L", image.size, 0)
    ImageDraw.Draw(mask).rounded_rectangle((0, 0, *image.size), radius=radius, fill=255)
    output = image.convert("RGBA")
    output.putalpha(mask)
    return output


def _gradient_image(width: int, height: int, start: str, end: str, diagonal: bool = True):
    """Build a diagonal linear-gradient RGB array from two hex colours."""
    import numpy as np
    from matplotlib.colors import to_rgb

    start_rgb = np.array(to_rgb(start))
    end_rgb = np.array(to_rgb(end))
    ys, xs = np.mgrid[0:height, 0:width]
    if diagonal:
        mix = (xs / max(1, width - 1) + ys / max(1, height - 1)) / 2
    else:
        mix = np.tile(xs / max(1, width - 1), (height, 1))
    return start_rgb[None, None, :] * (1 - mix[..., None]) + end_rgb[None, None, :] * mix[..., None]


def write_banner(path: Path) -> Path:
    """Hero banner: gradient panel, title, tagline and the three agent chips."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import FancyBboxPatch
    from PIL import Image

    width, height, scale = 1120, 420, 2
    fig = plt.figure(figsize=(width / 100, height / 100), dpi=100 * scale)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, width)
    ax.set_ylim(height, 0)
    ax.axis("off")

    # gradient background
    ax.imshow(
        _gradient_image(width, height, "#071C4A", "#0B6E4F"),
        extent=[0, width, height, 0],
        aspect="auto",
        zorder=0,
    )
    # faint grid
    for x in range(0, width, 34):
        ax.plot([x, x], [0, height], color=WHITE, alpha=0.05, linewidth=0.6, zorder=1)
    for y in range(0, height, 34):
        ax.plot([0, width], [y, y], color=WHITE, alpha=0.05, linewidth=0.6, zorder=1)
    # soft accent blobs
    for bx, by, br, colour, alpha in [
        (985, 55, 170, TEAL, 0.22),
        (95, 395, 190, "#4C8DFF", 0.16),
    ]:
        for step, step_alpha in ((1.0, alpha), (0.75, alpha * 0.7), (0.5, alpha * 0.5)):
            ax.add_patch(
                plt.Circle(
                    (bx, by),
                    br * step,
                    color=colour,
                    alpha=step_alpha * 0.45,
                    zorder=1,
                    linewidth=0,
                )
            )

    # candlestick motif
    candles = [70, 52, 84, 60, 96, 74, 108]
    for index, body in enumerate(candles):
        cx = 800 + index * 46
        top = 150 - body
        ax.plot(
            [cx, cx], [top - 16, top + 52], color="#8FE3C0", alpha=0.55, linewidth=1.6, zorder=2
        )
        colour = TEAL if index % 3 else AMBER
        ax.add_patch(
            FancyBboxPatch(
                (cx - 8, top),
                16,
                38,
                boxstyle="round,pad=0,rounding_size=3",
                linewidth=0,
                facecolor=colour,
                alpha=0.85,
                zorder=3,
            )
        )

    # copy
    ax.text(
        96,
        108,
        "MULTI-AGENT FINANCIAL RESEARCH",
        color="#8FE3C0",
        fontsize=12,
        weight="bold",
        family="DejaVu Sans",
        zorder=4,
    )
    ax.text(
        92,
        178,
        "Financial Analysis",
        color=WHITE,
        fontsize=40,
        weight="bold",
        family="DejaVu Sans",
        zorder=4,
    )
    ax.text(
        92,
        232,
        "Agent Crew",
        color="#8FE3C0",
        fontsize=40,
        weight="bold",
        family="DejaVu Sans",
        zorder=4,
    )
    ax.text(
        96,
        268,
        "agno · Gemini · Streamlit · yfinance · Plotly · reportlab",
        color="#C6D4E8",
        fontsize=12.5,
        family="DejaVu Sans",
        zorder=4,
    )
    ax.text(
        96,
        289,
        "Three coordinated agents with a deterministic revision loop",
        color="#9FB3CE",
        fontsize=11,
        family="DejaVu Sans",
        zorder=4,
    )

    # agent chips
    chips = [
        ("Market Researcher", "news & sentiment", TEAL),
        ("Data Analyst", "KPIs & charts", "#4C8DFF"),
        ("Report Writer", "report & PDF", AMBER),
    ]
    x = 96
    for name, subtitle, colour in chips:
        ax.add_patch(
            FancyBboxPatch(
                (x, 312),
                268,
                70,
                boxstyle="round,pad=0,rounding_size=14",
                linewidth=1,
                edgecolor=(1, 1, 1, 0.18),
                facecolor=(1, 1, 1, 0.07),
                zorder=3,
            )
        )
        ax.add_patch(plt.Circle((x + 26, 347), 7, color=colour, zorder=4))
        ax.add_patch(plt.Circle((x + 26, 347), 12, color=colour, alpha=0.22, zorder=4, linewidth=0))
        ax.text(
            x + 46,
            341,
            name,
            color=WHITE,
            fontsize=11.5,
            weight="bold",
            family="DejaVu Sans",
            zorder=4,
        )
        ax.text(
            x + 46, 362, subtitle, color="#B9C7DD", fontsize=9.5, family="DejaVu Sans", zorder=4
        )
        x += 288

    temp = path.with_name("_banner_raw.png")
    fig.savefig(temp, dpi=100 * scale, facecolor=fig.get_facecolor())
    plt.close(fig)
    # resize to the exact banner size and round the corners
    with Image.open(temp) as handle:
        banner = handle.convert("RGB").resize((width * scale, height * scale), Image.LANCZOS)
    temp.unlink(missing_ok=True)
    _rounded_alpha(banner, radius=44).save(path)
    return path


def write_architecture(path: Path) -> Path:
    """Pipeline diagram: inputs → orchestrator → three agents → outputs."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

    width, height, scale = 1120, 520, 2
    fig = plt.figure(figsize=(width / 100, height / 100), dpi=100 * scale)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, width)
    ax.set_ylim(height, 0)
    ax.axis("off")
    fig.patch.set_facecolor(WHITE)

    def box(x, y, w, h, facecolor, edgecolor="#D5DCE5", radius=12, linewidth=1.0):
        ax.add_patch(
            FancyBboxPatch(
                (x, y),
                w,
                h,
                boxstyle=f"round,pad=0,rounding_size={radius}",
                facecolor=facecolor,
                edgecolor=edgecolor,
                linewidth=linewidth,
                zorder=2,
            )
        )

    def arrow(x1, y1, x2, y2, colour="#7C8CA5", style="-|>", dashed=False, rad=0.0):
        ax.add_patch(
            FancyArrowPatch(
                (x1, y1),
                (x2, y2),
                arrowstyle=style,
                mutation_scale=13,
                linewidth=1.6,
                color=colour,
                linestyle="--" if dashed else "-",
                connectionstyle=f"arc3,rad={rad}",
                zorder=3,
            )
        )

    def label(x, y, text, size=11, weight="normal", colour="#1F2937", **kwargs):
        ax.text(
            x,
            y,
            text,
            fontsize=size,
            weight=weight,
            color=colour,
            family="DejaVu Sans",
            va="center",
            zorder=4,
            **kwargs,
        )

    # inputs
    box(24, 40, 180, 96, "#F7F9FD")
    label(40, 66, "INPUTS", 10, "bold", NAVY)
    label(40, 90, "Ticker + date range", 10, colour="#5B6472")
    label(40, 112, "Gemini key (optional)", 10, colour="#5B6472")
    arrow(204, 88, 258, 88)

    # orchestrator
    box(264, 26, 360, 124, "#0B3D91", edgecolor="#0B3D91", radius=14)
    ax.add_patch(
        FancyBboxPatch(
            (264, 26),
            360,
            124,
            boxstyle="round,pad=0,rounding_size=14",
            facecolor=(0.043, 0.431, 0.310, 0.55),
            edgecolor="none",
            zorder=2,
        )
    )
    label(284, 52, "ORCHESTRATOR", 10, "bold", "#B9E7D4")
    label(284, 80, "agno Team · coordinate mode", 13, "bold", WHITE)
    label(284, 106, "Delegates in order · verifies output", 10, colour="#D6E2F5")
    label(284, 128, "Revision loop (max N rounds)", 10, colour="#D6E2F5")

    # revision loop: writer -> orchestrator (dashed, amber)
    ax.add_patch(
        FancyArrowPatch(
            (700, 208),
            (622, 152),
            arrowstyle="-|>",
            mutation_scale=13,
            linewidth=1.6,
            color=AMBER,
            linestyle="--",
            connectionstyle="arc3,rad=-0.35",
            zorder=5,
        )
    )
    label(712, 168, "revision", 9.5, colour="#9A7420", style="italic")
    label(712, 182, "loop", 9.5, colour="#9A7420", style="italic")

    # agents
    agents = [
        (
            24,
            "1 · Market Researcher",
            "#0B6E4F",
            ["search_stock_news", "analyze_news_sentiment", "5-10 items · ±1 score · risks"],
        ),
        (
            292,
            "2 · Data Analyst",
            "#1F5FBF",
            [
                "get_stock_snapshot",
                "get_financial_kpis · charts",
                "P/E · ROI · RSI · MA · drawdown",
            ],
        ),
        (
            560,
            "3 · Report Writer",
            "#9A7420",
            [
                "no tools — pure synthesis",
                "7-section report contract",
                "label + confidence + disclaimer",
            ],
        ),
    ]
    for x, title, colour, lines in agents:
        box(x, 210, 248, 112, "#FAFBFE")
        label(x + 16, 240, title, 11.5, "bold", colour)
        label(x + 16, 264, lines[0], 10, colour="#5B6472")
        label(x + 16, 286, lines[1], 10, colour="#5B6472")
        label(x + 16, 308, lines[2], 8.8, colour="#8B95A5")

    arrow(272, 266, 288, 266)
    arrow(540, 266, 556, 266)
    arrow(808, 266, 834, 266)

    # orchestration fan-out
    for x in (148, 416, 684):
        arrow(444, 150, x, 206, colour="#B3BECD", rad=-0.08)

    # deterministic core
    box(24, 356, 784, 128, "#F7F9FD")
    label(40, 380, "DETERMINISTIC CORE  (NO LLM REQUIRED)", 10, "bold", NAVY)
    core = [
        ("finance_tools.py — yfinance OHLCV + 20 KPIs", 40, 412),
        ("sentiment.py — VADER + finance lexicon", 40, 436),
        ("chart_tools.py — Plotly charts exported to PNG", 40, 460),
        ("pdf_generator.py — reportlab, 7 sections", 470, 412),
        ("offline_crew.py — full crew without an LLM", 470, 436),
        ("logger.py — conversation log + rotating logs", 470, 460),
    ]
    for text, x, y in core:
        label(x, y, "• " + text, 9.6, colour="#3C4657")

    # outputs
    box(836, 210, 260, 274, "#0B1220", edgecolor="#0B1220", radius=14)
    label(856, 240, "OUTPUTS", 10, "bold", "#8FE3C0")
    outputs = [
        "PDF report (7 sections)",
        "4 interactive charts",
        "Markdown + KPI JSON",
        "Agent conversation log",
        "daily_summary.json (CLI)",
    ]
    for index, text in enumerate(outputs):
        ax.add_patch(plt.Circle((860, 266 + index * 26), 3, color="#8FE3C0", zorder=4))
        label(874, 266 + index * 26, text, 9.8, colour="#DCE6F5")
    ax.plot([856, 1076], [412, 412], color=WHITE, alpha=0.15, linewidth=1, zorder=3)
    label(856, 434, "Every artefact is written to", 8.8, colour="#8FA3BF")
    label(856, 451, "reports/output/ and survives", 8.8, colour="#8FA3BF")
    label(856, 468, "LLM or provider failures.", 8.8, colour="#8FA3BF")

    fig.savefig(path, dpi=100 * scale, facecolor=WHITE, bbox_inches=None)
    plt.close(fig)
    return path


# ---------------------------------------------------------------------------
# Animated demo GIF
# ---------------------------------------------------------------------------
@dataclass
class DemoFrame:
    """What the GIF should show at a given moment."""

    active_agent: int
    completed: tuple[bool, bool, bool]
    price_progress: float
    bars_progress: float
    show_pdf: bool
    status: str


def _demo_frames() -> list[DemoFrame]:
    """Build a deterministic 46-frame storyboard for one analysis run."""
    frames: list[DemoFrame] = []
    stages = [
        (0, "Fetching market data and recent news…"),
        (1, "Market Researcher is reading headlines…"),
        (2, "Data Analyst is computing KPIs…"),
        (3, "Report Writer is drafting the report…"),
        (4, "Reviewing the draft (revision loop)…"),
        (5, "Rendering the PDF report…"),
    ]
    # Stage 0: bootstrap
    for _ in range(6):
        frames.append(DemoFrame(0, (False, False, False), 0.0, 0.0, False, stages[0][1]))
    # Researcher
    for _ in range(7):
        frames.append(DemoFrame(0, (False, False, False), 0.0, 0.0, False, stages[1][1]))
    frames.append(DemoFrame(1, (True, False, False), 0.05, 0.0, False, stages[2][1]))
    # Analyst: the price line draws while the analyst computes KPIs
    for index in range(1, 15):
        frames.append(
            DemoFrame(
                1,
                (True, False, False),
                index / 14,
                index / 28,
                False,
                stages[2][1],
            )
        )
    frames.append(DemoFrame(2, (True, True, False), 1.0, 0.5, False, stages[3][1]))
    # Writer: the KPI bars grow while the report is drafted
    for index in range(1, 9):
        frames.append(DemoFrame(2, (True, True, False), 1.0, 0.5 + index / 16, False, stages[4][1]))
    frames.append(DemoFrame(3, (True, True, True), 1.0, 1.0, False, stages[5][1]))
    for _ in range(6):
        frames.append(
            DemoFrame(3, (True, True, True), 1.0, 1.0, True, "Report ready — PDF + charts saved")
        )
    return frames


def write_demo_gif(path: Path, fps: int = 12) -> Path:
    """Render the pipeline animation to an optimised GIF using real demo data."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import FancyBboxPatch, Rectangle
    from PIL import Image

    from agents.offline_crew import heuristic_recommendation
    from tools.demo_data import demo_news
    from tools.finance_tools import compute_kpis, fetch_stock_data
    from tools.sentiment import analyze_texts

    start, end = date(2024, 1, 2), date(2024, 9, 30)
    stock = fetch_stock_data("AAPL", start, end)
    kpis = compute_kpis(stock)
    sentiment = analyze_texts(demo_news("AAPL", stock.company_name, 6))
    verdict = heuristic_recommendation(kpis, sentiment.to_dict())
    prices = stock.history["Close"].to_numpy()

    fig = plt.figure(figsize=(9.6, 5.4), dpi=100)
    fig.patch.set_facecolor(INK)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    ax.set_facecolor(INK)

    agent_names = ["Market Researcher", "Data Analyst", "Report Writer"]

    def pill(
        x: float,
        y: float,
        w: float,
        h: float,
        colour: str,
        alpha: float = 1.0,
        radius: float = 0.02,
    ):
        patch = FancyBboxPatch(
            (x, y),
            w,
            h,
            boxstyle=f"round,pad=0,rounding_size={radius}",
            linewidth=0,
            facecolor=colour,
            alpha=alpha,
            transform=ax.transAxes,
            zorder=1,
        )
        ax.add_patch(patch)
        return patch

    def frame_to_image(frame: DemoFrame) -> Image.Image:
        ax.clear()
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis("off")

        # ---- header -------------------------------------------------------
        ax.text(
            0.035, 0.93, "Financial Analysis Agent Crew", color=WHITE, fontsize=15, weight="bold"
        )
        ax.text(
            0.035, 0.885, "AAPL  ·  2024-01-02 → 2024-09-30  ·  demo run", color=MUTED, fontsize=8.5
        )
        pill(0.815, 0.895, 0.15, 0.055, "#16233B", radius=0.012)
        ax.text(
            0.89, 0.9225, "● live pipeline", color="#8FE3C0", fontsize=8, ha="center", va="center"
        )
        ax.plot([0.035, 0.965], [0.855, 0.855], color="#1E2C46", linewidth=1)

        # ---- agent cards --------------------------------------------------
        colours = [TEAL, "#4C8DFF", AMBER]
        for index, name in enumerate(agent_names):
            x = 0.035 + index * 0.225
            done = frame.completed[index]
            active = frame.active_agent == index and not done
            border = colours[index] if (active or done) else "#22304A"
            pill(x, 0.60, 0.205, 0.20, "#101B2E", radius=0.014)
            ax.add_patch(
                FancyBboxPatch(
                    (x, 0.60),
                    0.205,
                    0.20,
                    boxstyle="round,pad=0,rounding_size=0.014",
                    linewidth=1.6 if (active or done) else 0.8,
                    edgecolor=border,
                    facecolor="none",
                    transform=ax.transAxes,
                    zorder=2,
                )
            )
            ax.text(x + 0.018, 0.765, f"{index + 1}", color=border, fontsize=11, weight="bold")
            ax.text(
                x + 0.043,
                0.765,
                name,
                color=WHITE if (active or done) else MUTED,
                fontsize=9.5,
                weight="bold",
            )
            subtitle = "✓ complete" if done else "working…" if active else "waiting"
            ax.text(
                x + 0.018,
                0.715,
                subtitle,
                color=colours[index] if (done or active) else "#5B6B85",
                fontsize=8,
            )
            # little progress ticks
            ticks = 3 if done else (2 if active else 0)
            for tick in range(ticks):
                pill(x + 0.018 + tick * 0.03, 0.645, 0.022, 0.012, colours[index], radius=0.004)
            if active:
                pill(
                    x + 0.018 + (ticks % 3) * 0.03,
                    0.645,
                    0.022,
                    0.012,
                    colours[index],
                    alpha=0.35,
                    radius=0.004,
                )
            if done:
                ax.text(x + 0.185, 0.615, "✓", color=colours[index], fontsize=13, ha="center")

        # ---- price chart --------------------------------------------------
        chart_x, chart_y, chart_w, chart_h = 0.035, 0.30, 0.44, 0.24
        pill(
            chart_x - 0.012,
            chart_y - 0.045,
            chart_w + 0.024,
            chart_h + 0.10,
            "#0E1729",
            radius=0.012,
        )
        ax.text(
            chart_x,
            chart_y + chart_h + 0.025,
            "Price & moving averages",
            color=WHITE,
            fontsize=8.5,
            weight="bold",
        )
        if frame.price_progress > 0:
            count = max(2, int(len(prices) * frame.price_progress))
            series = prices[:count]
            xs = [chart_x + chart_w * i / max(1, len(prices) - 1) for i in range(count)]
            low, high = float(prices.min()), float(prices.max())
            span = max(1e-6, high - low)
            ys = [chart_y + chart_h * (value - low) / span for value in series]
            ax.plot(xs, ys, color="#4C8DFF", linewidth=1.8)
            ax.fill_between(xs, ys, chart_y, color="#4C8DFF", alpha=0.15)
            # rolling mean overlay
            if count > 20:
                window = 20
                means = [series[i - window : i + 1].mean() for i in range(window, count)]
                mean_xs = xs[window:]
                mean_ys = [chart_y + chart_h * (value - low) / span for value in means]
                ax.plot(mean_xs, mean_ys, color=AMBER, linewidth=1.0, linestyle=":")
            ax.text(
                chart_x,
                chart_y - 0.022,
                f"{prices[count - 1]:,.2f} USD",
                color="#8FE3C0",
                fontsize=7.5,
            )

        # ---- KPI bars -----------------------------------------------------
        bar_x, bar_y, bar_w, bar_h = 0.53, 0.30, 0.435, 0.24
        pill(bar_x - 0.012, bar_y - 0.045, bar_w + 0.024, bar_h + 0.10, "#0E1729", radius=0.012)
        ax.text(
            bar_x,
            bar_y + bar_h + 0.025,
            "KPIs · ROI · volatility · RSI · P/E",
            color=WHITE,
            fontsize=8.5,
            weight="bold",
        )
        bars = [
            ("ROI%", kpis.get("period_roi_pct") or 0, TEAL),
            ("Vol%", kpis.get("volatility_annual_pct") or 0, "#4C8DFF"),
            ("DD%", kpis.get("max_drawdown_pct") or 0, RED),
            ("RSI", kpis.get("rsi_14") or 0, "#8FE3C0"),
            ("P/E", kpis.get("pe_ratio") or 0, AMBER),
        ]
        if frame.bars_progress > 0:
            scale = max(abs(value) for _, value, _ in bars) or 1.0
            slot = bar_w / len(bars)
            for index, (label, value, colour) in enumerate(bars):
                height = bar_h * 0.86 * abs(value) / scale * frame.bars_progress
                x = bar_x + index * slot + slot * 0.22
                ax.add_patch(
                    Rectangle(
                        (x, bar_y),
                        slot * 0.56,
                        height,
                        facecolor=colour,
                        edgecolor="none",
                        alpha=0.9,
                        transform=ax.transAxes,
                    )
                )
                ax.text(x + slot * 0.28, bar_y - 0.022, label, color=MUTED, fontsize=7, ha="center")
                if frame.bars_progress >= 0.95:
                    ax.text(
                        x + slot * 0.28,
                        bar_y + height + 0.012,
                        f"{value:,.1f}",
                        color=WHITE,
                        fontsize=6.5,
                        ha="center",
                    )

        # ---- status bar ---------------------------------------------------
        ax.plot([0.035, 0.965], [0.245, 0.245], color="#1E2C46", linewidth=1)
        dot_colour = "#8FE3C0" if frame.show_pdf else AMBER
        ax.plot([0.042], [0.205], marker="o", markersize=6, color=dot_colour)
        ax.text(0.055, 0.198, frame.status, color="#DCE6F5", fontsize=8.5)
        ax.text(
            0.965,
            0.198,
            f"sentiment {sentiment.label} ({sentiment.score:+.2f})  ·  "
            f"confidence {verdict['confidence']}/100",
            color=MUTED,
            fontsize=7.5,
            ha="right",
        )

        if frame.show_pdf:
            pill(0.035, 0.045, 0.30, 0.13, GREEN, radius=0.016)
            ax.text(
                0.185,
                0.122,
                "PDF REPORT READY",
                color=WHITE,
                fontsize=11,
                weight="bold",
                ha="center",
                va="center",
            )
            ax.text(
                0.185,
                0.082,
                "report + 4 charts + conversation log",
                color="#D8F3E6",
                fontsize=7.5,
                ha="center",
                va="center",
            )
            ax.text(
                0.965,
                0.122,
                f"recommendation  {verdict['recommendation']}",
                color=AMBER,
                fontsize=10.5,
                weight="bold",
                ha="right",
            )
            ax.text(
                0.965,
                0.092,
                "confidence  {}/100".format(verdict["confidence"]),
                color="#C8D4E6",
                fontsize=8.5,
                ha="right",
            )
        else:
            ax.text(0.965, 0.105, "streamlit run app.py", color="#3E4C66", fontsize=9, ha="right")

        fig.canvas.draw()
        return Image.frombuffer(
            "RGBA", fig.canvas.get_width_height(), fig.canvas.buffer_rgba(), "raw", "RGBA", 0, 1
        ).convert("RGB")

    frames = [frame_to_image(frame) for frame in _demo_frames()]
    plt.close(fig)

    # A shared palette keeps the GIF small without visible banding on these flat colours.
    palette_source = frames[-1].quantize(colors=128, method=Image.Quantize.MEDIANCUT)
    converted = [
        frame.quantize(palette=palette_source, dither=Image.Dither.NONE) for frame in frames
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    converted[0].save(
        path,
        save_all=True,
        append_images=converted[1:],
        duration=int(1000 / fps),
        loop=0,
        optimize=True,
        disposal=1,
    )
    return path


# ---------------------------------------------------------------------------
# Previews rendered from real output
# ---------------------------------------------------------------------------
def _shadowed_card(image, pad: int = 18, background=(246, 248, 252)):
    """Wrap a page render in a soft card with a subtle shadow."""
    from PIL import Image, ImageFilter

    width, height = image.size
    canvas = Image.new("RGB", (width + pad * 2, height + pad * 2), background)
    shadow = Image.new("RGBA", canvas.size, (0, 0, 0, 0))
    from PIL import ImageDraw

    draw = ImageDraw.Draw(shadow)
    draw.rounded_rectangle(
        (pad, pad + 6, pad + width, pad + height + 6), radius=10, fill=(15, 30, 60, 70)
    )
    shadow = shadow.filter(ImageFilter.GaussianBlur(10))
    canvas = Image.alpha_composite(canvas.convert("RGBA"), shadow)
    canvas.paste(image, (pad, pad))
    return canvas.convert("RGB")


def write_previews(asset_dir: Path, keep_pdf: bool) -> tuple[Path, Path]:
    """Render a real PDF and compose the README preview images from it."""
    import pymupdf
    from PIL import Image

    from agents.orchestrator import run_analysis

    print("  · running an offline analysis to produce a real report…")
    result = run_analysis("AAPL", date(2024, 1, 2), date(2024, 9, 30), use_llm=False)
    if not result.pdf_path:
        raise RuntimeError("PDF generation failed; cannot build previews")

    document = pymupdf.open(result.pdf_path)
    pages = []
    for page_index in (0, 2):  # cover + charts page
        pixmap = document[page_index].get_pixmap(dpi=110)
        pages.append(Image.frombytes("RGB", (pixmap.width, pixmap.height), pixmap.samples))

    cards = [_shadowed_card(page) for page in pages]
    gap = 24
    total_w = sum(card.width for card in cards) + gap
    total_h = max(card.height for card in cards)
    spread = Image.new("RGB", (total_w, total_h), (246, 248, 252))
    x = 0
    for card in cards:
        spread.paste(card, (x, (total_h - card.height) // 2))
        x += card.width + gap
    report_preview = asset_dir / "report-preview.png"
    spread.save(report_preview, optimize=True)

    # --- charts grid -------------------------------------------------------
    chart_paths = [result.chart_images.get(key) for key in ("price", "volume", "rsi", "kpi")]
    chart_paths = [Path(p) for p in chart_paths if p and Path(p).exists()]
    if len(chart_paths) < 4:
        raise RuntimeError("expected four chart images from the analysis run")

    tiles = []
    for chart_path in chart_paths:
        with Image.open(chart_path) as handle:
            tile = handle.convert("RGB")
            tile.thumbnail((760, 420), Image.LANCZOS)
            tiles.append(_shadowed_card(tile, pad=10, background=(255, 255, 255)))

    col_gap, row_gap = 16, 16
    tile_w = max(tile.width for tile in tiles)
    tile_h = max(tile.height for tile in tiles)
    grid = Image.new(
        "RGB",
        (tile_w * 2 + col_gap, tile_h * 2 + row_gap),
        (246, 248, 252),
    )
    for index, tile in enumerate(tiles):
        column, row = index % 2, index // 2
        grid.paste(tile, (column * (tile_w + col_gap), row * (tile_h + row_gap)))
    charts_preview = asset_dir / "charts-preview.png"
    grid.save(charts_preview, optimize=True)

    if not keep_pdf:
        for artifact in (result.pdf_path, result.report_path, result.log_path):
            if artifact and Path(artifact).exists():
                Path(artifact).unlink(missing_ok=True)
        for chart_path in result.chart_images.values():
            Path(chart_path).unlink(missing_ok=True)

    return report_preview, charts_preview


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def human_size(path: Path) -> str:
    size = path.stat().st_size
    if size > 1024 * 1024:
        return f"{size / (1024 * 1024):.2f} MB"
    return f"{size / 1024:.0f} KB"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Generate README assets (offline, deterministic).")
    parser.add_argument(
        "--keep-pdf", action="store_true", help="Keep the temporary PDF/report files"
    )
    parser.add_argument(
        "--skip-gif", action="store_true", help="Skip the animated GIF (slowest step)"
    )
    args = parser.parse_args(argv)

    ASSETS.mkdir(parents=True, exist_ok=True)
    generated: list[Path] = []

    print("Generating README assets…")
    generated.append(write_banner(ASSETS / "banner.png"))
    print("  ✓ banner.png")
    generated.append(write_architecture(ASSETS / "architecture.png"))
    print("  ✓ architecture.png")

    if not args.skip_gif:
        generated.append(write_demo_gif(ASSETS / "demo.gif"))
        print("  ✓ demo.gif")

    report_preview, charts_preview = write_previews(ASSETS, keep_pdf=args.keep_pdf)
    generated.extend([report_preview, charts_preview])
    print("  ✓ report-preview.png")
    print("  ✓ charts-preview.png")

    print("\nGenerated:")
    for path in generated:
        print(f"  {path.relative_to(ROOT)}  ({human_size(path)})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
