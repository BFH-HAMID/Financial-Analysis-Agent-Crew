"""PDF report generation with reportlab.

Layout
------
1. Cover block: ticker, company, analysis window, mode badges and headline metrics.
2. Executive summary, company snapshot, news & sentiment, KPI table, charts,
   risk assessment, recommendation + confidence, methodology, disclaimer.
3. Optional appendix: the full agent conversation log.

The renderer is defensive by design: it works from the Markdown report produced by
the crew (splitting it into sections with :mod:`agents.orchestrator` helpers) and
falls back to structured data when a section is missing. Charts are embedded from
the PNGs rendered in :mod:`tools.chart_tools`, so the PDF never depends on a live
Plotly/kaleido runtime.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from datetime import datetime
from pathlib import Path
from typing import Any

from config.settings import get_settings
from prompts.prompts import DISCLAIMER_TEXT, SECTION_ALIASES
from tools.finance_tools import build_kpi_table
from utils.formatting import fmt_compact, fmt_pct, sentiment_emoji
from utils.logger import get_logger

logger = get_logger(__name__)

from reportlab.lib import colors  # noqa: E402  (kept after logger for import clarity)
from reportlab.lib.enums import TA_LEFT  # noqa: E402
from reportlab.lib.pagesizes import A4  # noqa: E402
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet  # noqa: E402
from reportlab.lib.units import mm  # noqa: E402
from reportlab.lib.utils import ImageReader  # noqa: E402
from reportlab.platypus import (  # noqa: E402
    BaseDocTemplate,
    Frame,
    Image,
    KeepTogether,
    PageBreak,
    PageTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle,
)

PAGE_WIDTH, PAGE_HEIGHT = A4
MARGIN = 18 * mm

BRAND = colors.HexColor("#0B3D91")
ACCENT = colors.HexColor("#0B6E4F")
MUTED = colors.HexColor("#5B6472")
LIGHT = colors.HexColor("#EEF2F7")
GRID = colors.HexColor("#D5DCE5")

RECOMMENDATION_COLOURS = {
    "STRONG BUY": colors.HexColor("#0B6E4F"),
    "BUY": colors.HexColor("#2E8B57"),
    "HOLD": colors.HexColor("#B08900"),
    "SELL": colors.HexColor("#C0392B"),
    "STRONG SELL": colors.HexColor("#8E1B10"),
}

_REPLACEMENTS = {
    "→": "->",
    "←": "<-",
    "↑": "^",
    "↓": "v",
    "−": "-",
    "–": "-",
    "—": "-",
    "≈": "~",
    "≤": "<=",
    "≥": ">=",
    "×": "x",
    "√": "sqrt",
    "·": "-",
    "•": "-",
    "…": "...",
    "\u00a0": " ",
    "’": "'",
    "‘": "'",
    "“": '"',
    "”": '"',
}


# ---------------------------------------------------------------------------
# Text helpers
# ---------------------------------------------------------------------------
def sanitize_text(text: Any) -> str:
    """Make text safe for reportlab's built-in (Latin-1) fonts.

    Emoji and arrows are common in LLM output but missing from the standard
    fonts, so they are transliterated or removed instead of raising.
    """
    if text is None:
        return ""
    value = str(text)
    for source, target in _REPLACEMENTS.items():
        value = value.replace(source, target)
    return value.encode("latin-1", "ignore").decode("latin-1")


#: Bold ``**text**`` / ``__text__`` or code spans `` `text` ``.
_BOLD_CODE_PATTERN = re.compile(r"(\*\*.+?\*\*|__.+?__|`[^`\n]+?`)", re.DOTALL)
#: Italic ``*text*`` / ``_text_`` delimited by whitespace/brackets (so that
#: identifiers such as ``ALLOW_DEMO_DATA`` are never treated as italics).
#: Inline tags that may pass through untouched (unknown tags get escaped).
_ALLOWED_TAG_RE = re.compile(r"</?(?:b|i|u|font|br)\b[^<>]*>")

_ITALIC_PATTERN = re.compile(r"(^|[\s(\[])(\*[^*\s][^*\n]*?\*|_[^_\s][^_\n]*?_)(?=[\s.,;:)!?\]]|$)")


def _escape_preserving_tags(text: str) -> str:
    """Escape ``&``/``<``/``>`` while keeping a safe whitelist of inline tags."""
    text = text.replace("&", "&amp;")
    pieces: list[str] = []
    cursor = 0
    for match in _ALLOWED_TAG_RE.finditer(text):
        pieces.append(text[cursor : match.start()].replace("<", "&lt;").replace(">", "&gt;"))
        pieces.append(match.group(0))
        cursor = match.end()
    pieces.append(text[cursor:].replace("<", "&lt;").replace(">", "&gt;"))
    return "".join(pieces)


def inline_markup(text: str) -> str:
    """Convert a small Markdown subset (bold, italic, code) to reportlab XML.

    Inline tags such as ``<b>`` are preserved so callers may mix raw markup with
    Markdown. Underscores inside identifiers (``ALLOW_DEMO_DATA``) are never
    treated as italics, and unknown tags are escaped.
    """
    escaped = _escape_preserving_tags(sanitize_text(text))
    output: list[str] = []
    for chunk in _BOLD_CODE_PATTERN.split(escaped):
        if not chunk:
            continue
        if (chunk.startswith("**") and chunk.endswith("**")) or (
            chunk.startswith("__") and chunk.endswith("__")
        ):
            output.append(f"<b>{chunk[2:-2]}</b>")
        elif chunk.startswith("`") and chunk.endswith("`") and len(chunk) > 2:
            output.append(f'<font face="Courier">{chunk[1:-1]}</font>')
        else:
            output.append(
                _ITALIC_PATTERN.sub(lambda m: f"{m.group(1)}<i>{m.group(2)[1:-1]}</i>", chunk)
            )
    return "".join(output)


def split_sections(markdown: str) -> dict[str, str]:
    """Split Markdown into ``{lowercase heading: body}`` (duplicate of the orchestrator helper).

    Imported lazily inside functions elsewhere to avoid a circular import; the
    small duplication here keeps this module independently usable.
    """
    sections: dict[str, str] = {}
    pattern = re.compile(r"^(#{1,4})\s+(.*?)\s*$", re.MULTILINE)
    matches = list(pattern.finditer(markdown or ""))
    for index, match in enumerate(matches):
        heading = match.group(2).strip().lower()
        body_start = match.end()
        body_end = matches[index + 1].start() if index + 1 < len(matches) else len(markdown)
        sections[heading] = markdown[body_start:body_end].strip()
    return sections


def find_section(sections: dict[str, str], key: str) -> str | None:
    """Return the body of the first alias present for ``key``."""
    for alias in SECTION_ALIASES.get(key, (key,)):
        if alias in sections and sections[alias].strip():
            return sections[alias]
    return None


# ---------------------------------------------------------------------------
# Styles
# ---------------------------------------------------------------------------
def _build_styles() -> dict[str, ParagraphStyle]:
    styles = getSampleStyleSheet()
    base_font = "Helvetica"
    return {
        "title": ParagraphStyle(
            "CrewTitle",
            parent=styles["Title"],
            fontName=f"{base_font}-Bold",
            fontSize=23,
            leading=27,
            textColor=BRAND,
            alignment=TA_LEFT,
            spaceAfter=2,
        ),
        "subtitle": ParagraphStyle(
            "CrewSubtitle",
            parent=styles["Normal"],
            fontName=base_font,
            fontSize=10.5,
            leading=14,
            textColor=MUTED,
            spaceAfter=10,
        ),
        "h1": ParagraphStyle(
            "CrewH1",
            parent=styles["Heading1"],
            fontName=f"{base_font}-Bold",
            fontSize=15,
            leading=19,
            textColor=BRAND,
            spaceBefore=12,
            spaceAfter=6,
        ),
        "h2": ParagraphStyle(
            "CrewH2",
            parent=styles["Heading2"],
            fontName=f"{base_font}-Bold",
            fontSize=12,
            leading=15,
            textColor=ACCENT,
            spaceBefore=10,
            spaceAfter=4,
        ),
        "h3": ParagraphStyle(
            "CrewH3",
            parent=styles["Heading3"],
            fontName=f"{base_font}-Bold",
            fontSize=10.5,
            leading=13,
            textColor=colors.HexColor("#27313F"),
            spaceBefore=8,
            spaceAfter=3,
        ),
        "body": ParagraphStyle(
            "CrewBody",
            parent=styles["BodyText"],
            fontName=base_font,
            fontSize=9.4,
            leading=13.2,
            textColor=colors.HexColor("#1F2937"),
            spaceAfter=5,
        ),
        "bullet": ParagraphStyle(
            "CrewBullet",
            parent=styles["BodyText"],
            fontName=base_font,
            fontSize=9.4,
            leading=13,
            leftIndent=11,
            bulletIndent=2,
            spaceAfter=2.5,
        ),
        "small": ParagraphStyle(
            "CrewSmall",
            parent=styles["Normal"],
            fontName=base_font,
            fontSize=8,
            leading=10.5,
            textColor=MUTED,
        ),
        "caption": ParagraphStyle(
            "CrewCaption",
            parent=styles["Normal"],
            fontName=f"{base_font}-Oblique",
            fontSize=8,
            leading=10,
            textColor=MUTED,
            spaceBefore=2,
            spaceAfter=8,
        ),
        "table_head": ParagraphStyle(
            "CrewTableHead",
            parent=styles["Normal"],
            fontName=f"{base_font}-Bold",
            fontSize=8.3,
            leading=10,
            textColor=colors.white,
        ),
        "table_cell": ParagraphStyle(
            "CrewTableCell",
            parent=styles["Normal"],
            fontName=base_font,
            fontSize=8.3,
            leading=10,
            textColor=colors.HexColor("#1F2937"),
        ),
        "badge": ParagraphStyle(
            "CrewBadge",
            parent=styles["Normal"],
            fontName=f"{base_font}-Bold",
            fontSize=16,
            leading=20,
            textColor=colors.white,
            alignment=1,
        ),
    }


# ---------------------------------------------------------------------------
# Markdown -> flowables
# ---------------------------------------------------------------------------
def markdown_to_flowables(markdown: str, styles: dict[str, ParagraphStyle]) -> list[Any]:
    """Convert a Markdown fragment into reportlab flowables.

    Supports headings, paragraphs, bullet/numbered lists, pipe tables, block
    quotes and horizontal rules. Unknown syntax degrades to a paragraph.
    """
    flowables: list[Any] = []
    lines = (markdown or "").splitlines()
    index = 0

    while index < len(lines):
        line = lines[index].rstrip()
        stripped = line.strip()

        if not stripped:
            index += 1
            continue

        if stripped in {"---", "***", "___"}:
            flowables.append(Spacer(1, 3))
            flowables.append(
                Table(
                    [[""]],
                    colWidths=[PAGE_WIDTH - 2 * MARGIN],
                    rowHeights=[0.6],
                    style=TableStyle([("BACKGROUND", (0, 0), (-1, -1), GRID)]),
                )
            )
            flowables.append(Spacer(1, 5))
            index += 1
            continue

        heading = re.match(r"^(#{1,4})\s+(.*)$", stripped)
        if heading:
            level = min(len(heading.group(1)), 3)
            style = styles["h1"] if level == 1 else styles["h2"] if level == 2 else styles["h3"]
            flowables.append(Paragraph(inline_markup(heading.group(2)), style))
            index += 1
            continue

        if stripped.startswith(">"):
            quote_lines = []
            while index < len(lines) and lines[index].strip().startswith(">"):
                quote_lines.append(lines[index].strip().lstrip(">").strip())
                index += 1
            flowables.append(_callout(inline_markup(" ".join(quote_lines)), styles))
            continue

        if _is_table_row(stripped):
            table_lines = []
            while index < len(lines) and _is_table_row(lines[index].strip()):
                table_lines.append(lines[index].strip())
                index += 1
            table = _pipe_table(table_lines, styles)
            if table is not None:
                flowables.append(table)
                flowables.append(Spacer(1, 6))
            continue

        bullet_match = re.match(r"^[-*+]\s+(.*)$", stripped)
        numbered_match = re.match(r"^(\d+)[.)]\s+(.*)$", stripped)
        if bullet_match or numbered_match:
            text = bullet_match.group(1) if bullet_match else numbered_match.group(2)  # type: ignore[index]
            marker = "-" if bullet_match else f"{numbered_match.group(1)}."  # type: ignore[index]
            flowables.append(
                Paragraph(inline_markup(_join_lines([text])), styles["bullet"], bulletText=marker)
            )
            index += 1
            continue

        paragraph_lines = [stripped]
        index += 1
        while index < len(lines):
            candidate = lines[index].strip()
            if (
                not candidate
                or candidate.startswith("#")
                or candidate.startswith(">")
                or candidate in {"---", "***", "___"}
                or _is_table_row(candidate)
                or re.match(r"^[-*+]\s+", candidate)
                or re.match(r"^\d+[.)]\s+", candidate)
            ):
                break
            paragraph_lines.append(candidate)
            index += 1
        flowables.append(Paragraph(inline_markup(_join_lines(paragraph_lines)), styles["body"]))

    return flowables


def _join_lines(lines: Sequence[str]) -> str:
    """Join hard-wrapped Markdown lines into one paragraph."""
    return " ".join(line.strip() for line in lines if line.strip())


def _is_table_row(line: str) -> bool:
    return line.startswith("|") and line.endswith("|") and line.count("|") >= 2


def _pipe_table(lines: Sequence[str], styles: dict[str, ParagraphStyle]) -> Table | None:
    """Build a styled table from Markdown pipe rows."""
    rows: list[list[str]] = []
    for line in lines:
        cells = [cell.strip() for cell in line.strip().strip("|").split("|")]
        if all(re.fullmatch(r":?-{2,}:?", cell) for cell in cells if cell):
            continue  # separator row
        rows.append(cells)
    if not rows:
        return None

    column_count = max(len(row) for row in rows)
    rows = [row + [""] * (column_count - len(row)) for row in rows]
    data = [
        [
            Paragraph(inline_markup(cell), styles["table_head" if index == 0 else "table_cell"])
            for cell in row
        ]
        for index, row in enumerate(rows)
    ]

    available = PAGE_WIDTH - 2 * MARGIN
    if column_count == 3:
        widths = [available * 0.34, available * 0.24, available * 0.42]
    else:
        widths = [available / column_count] * column_count

    table = Table(data, colWidths=widths, repeatRows=1, hAlign="LEFT")
    table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), BRAND),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("GRID", (0, 0), (-1, -1), 0.4, GRID),
                ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, LIGHT]),
                ("LEFTPADDING", (0, 0), (-1, -1), 4),
                ("RIGHTPADDING", (0, 0), (-1, -1), 4),
                ("TOPPADDING", (0, 0), (-1, -1), 3),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
            ]
        )
    )
    return table


def _callout(text: str, styles: dict[str, ParagraphStyle], colour: colors.Color = BRAND) -> Table:
    """A soft coloured callout box around a paragraph."""
    table = Table(
        [[Paragraph(text, styles["body"])]],
        colWidths=[PAGE_WIDTH - 2 * MARGIN],
    )
    table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, -1), LIGHT),
                ("LINEBEFORE", (0, 0), (0, -1), 3, colour),
                ("LEFTPADDING", (0, 0), (-1, -1), 8),
                ("RIGHTPADDING", (0, 0), (-1, -1), 8),
                ("TOPPADDING", (0, 0), (-1, -1), 6),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
            ]
        )
    )
    return table


# ---------------------------------------------------------------------------
# Cover & section builders
# ---------------------------------------------------------------------------
def _cover(result: Any, styles: dict[str, ParagraphStyle], chart_count: int) -> list[Any]:
    kpis = getattr(result, "kpis", {}) or {}
    recommendation = getattr(result, "recommendation", None) or "HOLD"
    confidence = getattr(result, "confidence", None)
    colour = RECOMMENDATION_COLOURS.get(recommendation, MUTED)

    story: list[Any] = [Spacer(1, 6)]
    story.append(
        Paragraph(
            inline_markup(f"**{getattr(result, 'ticker', '')}** — Financial Analysis Report"),
            styles["title"],
        )
    )
    company = getattr(result, "company_name", "") or ""
    story.append(
        Paragraph(
            inline_markup(
                f"{company} · Analysis window {getattr(result, 'start', '')} to {getattr(result, 'end', '')} "
                f"· Generated {datetime.now().strftime('%d %b %Y, %H:%M')}"
            ),
            styles["subtitle"],
        )
    )

    badges = [
        f"Mode: {'Live agents (Gemini)' if getattr(result, 'llm_used', False) else 'Offline crew'}"
    ]
    badges.append(
        f"Data: {'SYNTHETIC DEMO' if getattr(result, 'is_demo', False) else str(kpis.get('source', 'yfinance'))}"
    )
    badges.append(f"Revision rounds: {getattr(result, 'revision_rounds', 0)}")
    badges.append(f"Charts: {chart_count}")
    story.append(Paragraph(inline_markup(" | ".join(badges)), styles["small"]))
    story.append(Spacer(1, 8))

    if getattr(result, "is_demo", False):
        story.append(
            _callout(
                "<b>SYNTHETIC DEMO DATA.</b> Live market and news providers were unreachable (or "
                "demo mode was forced). Every number in this document is simulated for product "
                "demonstration and must not be used for investment decisions.",
                styles,
                colors.HexColor("#C0392B"),
            )
        )
        story.append(Spacer(1, 8))

    # Company fact strip: profile data that no longer has its own section.
    facts = [
        ("Sector", str(kpis.get("sector") or "n/a")),
        ("Industry", str(kpis.get("industry") or "n/a")),
        ("Exchange", str(kpis.get("exchange") or "n/a")),
        (
            "Market cap",
            fmt_compact(
                kpis.get("market_cap"),
                symbol="$" if (kpis.get("currency") or "USD") == "USD" else "",
            ),
        ),
        (
            "52-week range",
            f"{_fmt_money(kpis.get('low_52w'), kpis.get('currency'))} - "
            f"{_fmt_money(kpis.get('high_52w'), kpis.get('currency'))}",
        ),
        ("Trading days", str(kpis.get("trading_days", "n/a"))),
    ]
    fact_cells = [
        [
            Paragraph(f"<b>{label}</b>", styles["small"]),
            Paragraph(inline_markup(value), styles["small"]),
        ]
        for label, value in facts
    ]
    fact_table = Table(fact_cells, colWidths=[26 * mm, 128 * mm])
    fact_table.setStyle(
        TableStyle(
            [
                ("GRID", (0, 0), (-1, -1), 0.3, GRID),
                ("BACKGROUND", (0, 0), (0, -1), LIGHT),
                ("TOPPADDING", (0, 0), (-1, -1), 2),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 2),
                ("LEFTPADDING", (0, 0), (-1, -1), 4),
            ]
        )
    )
    story.append(fact_table)
    story.append(Spacer(1, 8))

    metrics = [
        ("Last price", _fmt_money(kpis.get("current_price"), kpis.get("currency"))),
        ("Period ROI", fmt_pct(kpis.get("period_roi_pct"))),
        ("Volatility (ann.)", fmt_pct(kpis.get("volatility_annual_pct"), signed=False)),
        ("RSI (14)", _fmt_num(kpis.get("rsi_14"), 1)),
        ("P/E", _fmt_num(kpis.get("pe_ratio"))),
        ("Sharpe", _fmt_num(kpis.get("sharpe_ratio"))),
    ]
    metric_cells = [
        [
            Paragraph(f"<b>{label}</b>", styles["small"]),
            Paragraph(inline_markup(value), styles["body"]),
        ]
        for label, value in metrics
    ]
    metric_table = Table(metric_cells, colWidths=[30 * mm, 45 * mm])
    metric_table.setStyle(
        TableStyle(
            [
                ("GRID", (0, 0), (-1, -1), 0.3, GRID),
                ("BACKGROUND", (0, 0), (0, -1), LIGHT),
                ("TOPPADDING", (0, 0), (-1, -1), 2.5),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 2.5),
            ]
        )
    )

    recommendation_block = Table(
        [
            [Paragraph(f"RECOMMENDATION: {sanitize_text(recommendation)}", styles["badge"])],
            [
                Paragraph(
                    inline_markup(
                        f"Confidence: <b>{confidence if confidence is not None else 'n/a'}/100</b>"
                    ),
                    ParagraphStyle(
                        "badgeSub", parent=styles["small"], textColor=colors.white, alignment=1
                    ),
                )
            ],
        ],
        colWidths=[70 * mm],
    )
    recommendation_block.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, -1), colour),
                ("TOPPADDING", (0, 0), (-1, -1), 6),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
            ]
        )
    )

    hero = Table([[metric_table, recommendation_block]], colWidths=[80 * mm, 74 * mm])
    hero.setStyle(
        TableStyle(
            [
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("LEFTPADDING", (0, 0), (-1, -1), 0),
                ("RIGHTPADDING", (0, 0), (-1, -1), 4),
            ]
        )
    )
    story.append(hero)
    story.append(Spacer(1, 10))

    warning_lines = [
        warning
        for warning in (getattr(result, "warnings", []) or [])
        if not str(warning).startswith("Set ALLOW_DEMO_DATA")
        and "SYNTHETIC DEMO DATA" not in str(warning).upper()
    ][:4]
    deduped: list[str] = []
    for warning in warning_lines:
        text = str(warning)
        if "SYNTHETIC DEMO DATA" in text.upper() and any(
            "SYNTHETIC DEMO DATA" in existing.upper() for existing in deduped
        ):
            continue
        deduped.append(text)
    warning_lines = deduped
    if warning_lines:
        story.append(Paragraph("<b>Run notes</b>", styles["h3"]))
        for warning in warning_lines:
            story.append(Paragraph(inline_markup(warning), styles["bullet"], bulletText="-"))
    story.append(Spacer(1, 6))
    return story


def _fmt_money(value: Any, currency: str | None = None) -> str:
    from utils.formatting import fmt_currency

    symbol = "$" if (currency or "USD") == "USD" else f"{currency} "
    return fmt_currency(value, symbol=symbol)


def _fmt_num(value: Any, decimals: int = 2) -> str:
    from utils.formatting import fmt_number

    return fmt_number(value, decimals)


def _section(
    title: str, body: str | None, styles: dict[str, ParagraphStyle], fallback: str = ""
) -> list[Any]:
    """Render one titled section from Markdown, or a fallback sentence."""
    content = (body or "").strip() or fallback
    if not content:
        return []
    return [
        Paragraph(inline_markup(title), styles["h1"]),
        *markdown_to_flowables(content, styles),
    ]


def _kpi_table(result: Any, styles: dict[str, ParagraphStyle]) -> list[Any]:
    rows = build_kpi_table(getattr(result, "kpis", {}) or {})
    data = [
        [
            Paragraph("<b>Metric</b>", styles["table_head"]),
            Paragraph("<b>Value</b>", styles["table_head"]),
            Paragraph("<b>Note</b>", styles["table_head"]),
        ]
    ]
    for row in rows:
        data.append(
            [
                Paragraph(inline_markup(row["metric"]), styles["table_cell"]),
                Paragraph(inline_markup(row["value"]), styles["table_cell"]),
                Paragraph(inline_markup(row["note"]), styles["table_cell"]),
            ]
        )
    available = PAGE_WIDTH - 2 * MARGIN
    table = Table(
        data,
        colWidths=[available * 0.32, available * 0.26, available * 0.42],
        repeatRows=1,
        hAlign="LEFT",
    )
    table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), BRAND),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("GRID", (0, 0), (-1, -1), 0.4, GRID),
                ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, LIGHT]),
                ("LEFTPADDING", (0, 0), (-1, -1), 4),
                ("RIGHTPADDING", (0, 0), (-1, -1), 4),
                ("TOPPADDING", (0, 0), (-1, -1), 2.5),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 2.5),
            ]
        )
    )
    return [Paragraph("KPI Table", styles["h1"]), table, Spacer(1, 8)]


CHART_MAX_HEIGHT_MM = {"price": 78, "volume": 46, "rsi": 46, "kpi": 62}

CHART_CAPTIONS = {
    "price": "Price history with SMA20 / SMA50 / SMA200 overlays.",
    "volume": "Trading volume; bars are coloured by up/down days.",
    "rsi": "RSI(14) with 30/70 bands — overbought above 70, oversold below 30.",
    "kpi": "KPI bar chart mixing return, volatility, drawdown, RSI, P/E, Sharpe and beta.",
}


def _chart_flowables(result: Any, styles: dict[str, ParagraphStyle]) -> list[Any]:
    """Embed the rendered chart PNGs, keeping each image with its caption.

    The section heading is bundled with the first image so it never lands alone
    at the bottom of a page.
    """
    images: dict[str, str] = getattr(result, "chart_images", {}) or {}
    if not images:
        return []
    max_width = PAGE_WIDTH - 2 * MARGIN
    heading = Paragraph("Charts", styles["h1"])
    story: list[Any] = []
    for key in ("price", "volume", "rsi", "kpi"):
        path = images.get(key)
        if not path or not Path(path).exists():
            continue
        try:
            reader = ImageReader(path)
            width_px, height_px = reader.getSize()
            scale = max_width / float(width_px)
            height = height_px * scale
            max_height = CHART_MAX_HEIGHT_MM.get(key, 80) * mm
            if height > max_height:
                height = max_height
                width = width_px * (max_height / height_px)
            else:
                width = max_width
            block = [
                Image(path, width=width, height=height),
                Paragraph(inline_markup(CHART_CAPTIONS.get(key, key)), styles["caption"]),
            ]
            story.append(KeepTogether([heading, *block] if not story else block))
        except Exception as exc:  # noqa: BLE001
            logger.warning("Could not embed chart %s: %s", path, exc)
    if story:
        story.append(Spacer(1, 4))
    return story


def _news_flowables(result: Any, styles: dict[str, ParagraphStyle]) -> list[Any]:
    news: list[dict[str, Any]] = getattr(result, "news", []) or []
    if not news:
        return []
    data = [
        [
            Paragraph("<b>Headline</b>", styles["table_head"]),
            Paragraph("<b>Source</b>", styles["table_head"]),
            Paragraph("<b>Date</b>", styles["table_head"]),
        ]
    ]
    for item in news[:8]:
        title = str(item.get("title") or "")
        if item.get("is_demo"):
            title += " [synthetic]"
        data.append(
            [
                Paragraph(inline_markup(title), styles["table_cell"]),
                Paragraph(inline_markup(str(item.get("source") or "-")), styles["table_cell"]),
                Paragraph(inline_markup(str(item.get("published") or "-")), styles["table_cell"]),
            ]
        )
    available = PAGE_WIDTH - 2 * MARGIN
    table = Table(
        data,
        colWidths=[available * 0.62, available * 0.22, available * 0.16],
        repeatRows=1,
        hAlign="LEFT",
    )
    table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), BRAND),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("GRID", (0, 0), (-1, -1), 0.4, GRID),
                ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, LIGHT]),
                ("TOPPADDING", (0, 0), (-1, -1), 2.5),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 2.5),
            ]
        )
    )
    return [Paragraph("Headlines analysed", styles["h2"]), table, Spacer(1, 6)]


def _sentiment_box(result: Any, styles: dict[str, ParagraphStyle]) -> list[Any]:
    sentiment = getattr(result, "sentiment", {}) or {}
    if not sentiment:
        return []
    label = sentiment.get("label", "neutral")
    score = sentiment.get("score", 0.0)
    summary = (
        f"{sentiment_emoji(label)} <b>{str(label).title()}</b> — average score "
        f"<b>{float(score):+.2f}</b> on a -1 to +1 scale across "
        f"{sentiment.get('articles_analyzed', 0)} headlines "
        f"({sentiment.get('positive_count', 0)} positive / "
        f"{sentiment.get('neutral_count', 0)} neutral / "
        f"{sentiment.get('negative_count', 0)} negative)."
    )
    colour = (
        ACCENT
        if score >= 0.15
        else colors.HexColor("#C0392B")
        if score <= -0.15
        else colors.HexColor("#B08900")
    )
    flowables = [_callout(summary, styles, colour)]
    for observation in (sentiment.get("observations") or [])[:3]:
        flowables.append(Paragraph(inline_markup(observation), styles["bullet"], bulletText="-"))
    flowables.append(Spacer(1, 4))
    return flowables


_CONFIDENCE_BLOCK_RE = re.compile(r"\n\s*(?:\*\*)?confidence", re.IGNORECASE)

_LABEL_RE = re.compile(
    r"^\s*(?:\*\*|__)?\s*(STRONG\s+BUY|STRONG\s+SELL|BUY|SELL|HOLD)\b\s*(?:\*\*|__)?\s*[—–:\-]?\s*",
    re.IGNORECASE,
)


def _recommendation_flowables(
    result: Any, sections: dict[str, str], styles: dict[str, ParagraphStyle]
) -> list[Any]:
    """Render the recommendation section with a fixed-width label column."""
    body = find_section(sections, "recommendation")
    recommendation = getattr(result, "recommendation", None)
    if not body and not recommendation:
        return []

    flowables: list[Any] = [Paragraph("Recommendation", styles["h1"])]
    label = recommendation
    remainder = body or ""

    match = _LABEL_RE.match(remainder) if remainder else None
    if match:
        label = match.group(1).upper()
        remainder = remainder[match.end() :]

    # The confidence score gets its own block right below this section, so drop it
    # from the justification text to avoid printing the same rationale twice.
    remainder = _CONFIDENCE_BLOCK_RE.split(remainder, maxsplit=1)[0].strip()

    if label:
        colour = RECOMMENDATION_COLOURS.get(label.upper(), MUTED)
        badge = Table(
            [
                [
                    Paragraph(
                        sanitize_text(label.upper()),
                        ParagraphStyle(
                            "labelBadge", parent=styles["badge"], fontSize=13, leading=16
                        ),
                    )
                ]
            ],
            colWidths=[30 * mm],
        )
        badge.setStyle(
            TableStyle(
                [
                    ("BACKGROUND", (0, 0), (-1, -1), colour),
                    ("TOPPADDING", (0, 0), (-1, -1), 4),
                    ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
                ]
            )
        )
        right = [
            *(
                markdown_to_flowables(remainder, styles)
                if remainder.strip()
                else [Paragraph("No written justification was produced.", styles["body"])]
            ),
        ]
        wrapper = Table([[badge, right]], colWidths=[34 * mm, PAGE_WIDTH - 2 * MARGIN - 34 * mm])
        wrapper.setStyle(
            TableStyle(
                [
                    ("VALIGN", (0, 0), (-1, -1), "TOP"),
                    ("LEFTPADDING", (0, 0), (-1, -1), 0),
                    ("RIGHTPADDING", (0, 0), (-1, -1), 0),
                    ("TOPPADDING", (0, 0), (-1, -1), 0),
                    ("BOTTOMPADDING", (0, 0), (-1, -1), 0),
                ]
            )
        )
        flowables.append(wrapper)
        flowables.append(Spacer(1, 4))
    elif remainder:
        flowables.extend(markdown_to_flowables(remainder, styles))
    return flowables


def _confidence_flowables(result: Any, styles: dict[str, ParagraphStyle]) -> list[Any]:
    confidence = getattr(result, "confidence", None)
    if confidence is None:
        return []
    try:
        value = max(0, min(100, int(confidence)))
    except (TypeError, ValueError):
        return []

    label = "high" if value >= 75 else "moderate" if value >= 55 else "low"
    bar_width = (PAGE_WIDTH - 2 * MARGIN) * (value / 100)
    filled = Table([[""]], colWidths=[bar_width], rowHeights=[7 * mm])
    filled.setStyle(TableStyle([("BACKGROUND", (0, 0), (-1, -1), ACCENT)]))
    empty_width = (PAGE_WIDTH - 2 * MARGIN) - bar_width
    if empty_width > 1:
        empty = Table([[""]], colWidths=[empty_width], rowHeights=[7 * mm])
        empty.setStyle(TableStyle([("BACKGROUND", (0, 0), (-1, -1), LIGHT)]))
        bar = Table([[filled, empty]], colWidths=[bar_width, empty_width])
        bar.setStyle(
            TableStyle(
                [
                    ("LEFTPADDING", (0, 0), (-1, -1), 0),
                    ("RIGHTPADDING", (0, 0), (-1, -1), 0),
                    ("TOPPADDING", (0, 0), (-1, -1), 0),
                    ("BOTTOMPADDING", (0, 0), (-1, -1), 0),
                ]
            )
        )
    else:
        bar = filled

    flowables: list[Any] = [
        Paragraph("Confidence Score", styles["h1"]),
        Paragraph(
            inline_markup(f"<b>{value}/100</b> — {label} confidence in the recommendation."),
            styles["body"],
        ),
        bar,
        Spacer(1, 6),
    ]
    rationale = getattr(result, "confidence_rationale", None)
    if rationale:
        flowables.extend(markdown_to_flowables(str(rationale), styles))
    return flowables


def _conversation_flowables(conversation: Any, styles: dict[str, ParagraphStyle]) -> list[Any]:
    events = []
    if conversation is None:
        return []
    if hasattr(conversation, "events"):
        events = [event.to_dict() for event in conversation.events]
    elif isinstance(conversation, list):
        events = conversation
    if not events:
        return []

    flowables: list[Any] = [
        PageBreak(),
        Paragraph("Appendix — Agent Conversation Log", styles["h1"]),
    ]
    flowables.append(
        Paragraph(
            inline_markup(f"{len(events)} events recorded across the crew. Times are UTC."),
            styles["small"],
        )
    )
    flowables.append(Spacer(1, 4))

    data = [
        [
            Paragraph("<b>Time</b>", styles["table_head"]),
            Paragraph("<b>Agent</b>", styles["table_head"]),
            Paragraph("<b>Event</b>", styles["table_head"]),
            Paragraph("<b>Detail</b>", styles["table_head"]),
        ]
    ]
    for event in events[:120]:
        data.append(
            [
                Paragraph(inline_markup(str(event.get("timestamp", ""))), styles["table_cell"]),
                Paragraph(inline_markup(str(event.get("agent", ""))), styles["table_cell"]),
                Paragraph(inline_markup(str(event.get("event", ""))), styles["table_cell"]),
                Paragraph(inline_markup(str(event.get("message", ""))[:220]), styles["table_cell"]),
            ]
        )
    available = PAGE_WIDTH - 2 * MARGIN
    table = Table(
        data,
        colWidths=[available * 0.12, available * 0.18, available * 0.12, available * 0.58],
        repeatRows=1,
        hAlign="LEFT",
    )
    table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), BRAND),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("GRID", (0, 0), (-1, -1), 0.3, GRID),
                ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, LIGHT]),
                ("TOPPADDING", (0, 0), (-1, -1), 2),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 2),
            ]
        )
    )
    flowables.append(table)
    return flowables


# ---------------------------------------------------------------------------
# Page decorations
# ---------------------------------------------------------------------------
def _make_page_callbacks(app_name: str, footer_note: str):
    def on_page(canvas: Any, doc: Any) -> None:
        canvas.saveState()
        canvas.setFillColor(BRAND)
        canvas.rect(0, PAGE_HEIGHT - 12 * mm, PAGE_WIDTH, 12 * mm, stroke=0, fill=1)
        canvas.setFillColor(colors.white)
        canvas.setFont("Helvetica-Bold", 8.5)
        canvas.drawString(MARGIN, PAGE_HEIGHT - 8 * mm, sanitize_text(app_name))
        canvas.setFont("Helvetica", 7.5)
        canvas.drawRightString(
            PAGE_WIDTH - MARGIN,
            PAGE_HEIGHT - 8 * mm,
            sanitize_text(footer_note),
        )

        canvas.setStrokeColor(GRID)
        canvas.setLineWidth(0.5)
        canvas.line(MARGIN, 14 * mm, PAGE_WIDTH - MARGIN, 14 * mm)
        canvas.setFillColor(MUTED)
        canvas.setFont("Helvetica", 7.5)
        canvas.drawString(MARGIN, 10 * mm, "Automated research — not investment advice.")
        canvas.drawRightString(PAGE_WIDTH - MARGIN, 10 * mm, f"Page {doc.page}")
        canvas.restoreState()

    return on_page


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
def build_pdf_story(
    result: Any,
    conversation: Any = None,
    styles: dict[str, ParagraphStyle] | None = None,
    include_conversation: bool = True,
) -> list[Any]:
    """Assemble the full flowable story for one analysis.

    The order mirrors the Report Writer contract so the PDF reads like the
    Markdown report: Executive Summary, Market News Summary, KPI Table and
    Charts, Sentiment Overview, Analysis and Outlook, Recommendation (with the
    confidence score), Methodology, Disclaimer.
    """
    styles = styles or _build_styles()
    report_markdown = getattr(result, "report_markdown", "") or ""
    sections = split_sections(report_markdown)
    chart_images = getattr(result, "chart_images", {}) or {}

    story: list[Any] = []
    story.extend(_cover(result, styles, len(chart_images)))

    # 1 -------------------------------------------------- Executive Summary
    story.extend(
        _section(
            "Executive Summary",
            find_section(sections, "executive_summary"),
            styles,
            fallback="No executive summary was produced for this run.",
        )
    )

    # 2 ------------------------------------------------ Market News Summary
    news_body = find_section(sections, "news_summary")
    story.append(Paragraph("Market News Summary", styles["h1"]))
    if news_body:
        story.extend(markdown_to_flowables(news_body, styles))
    else:
        story.append(
            Paragraph(
                inline_markup("No news summary was produced; see the headline table below."),
                styles["body"],
            )
        )
    story.extend(_news_flowables(result, styles))

    # 3 ----------------------------------------------- KPI Table and Charts
    story.extend(_kpi_table(result, styles))
    charts = _chart_flowables(result, styles)
    if charts:
        story.extend(charts)
    else:
        chart_text = find_section(sections, "charts")
        if chart_text:
            story.extend(_section("Charts", chart_text, styles))

    # 4 -------------------------------------------------- Sentiment Overview
    sentiment_body = find_section(sections, "sentiment")
    story.append(Paragraph("Sentiment Overview", styles["h1"]))
    if sentiment_body:
        story.extend(markdown_to_flowables(sentiment_body, styles))
    else:
        story.extend(_sentiment_box(result, styles))

    # 5 ----------------------------------------------- Analysis and Outlook
    story.extend(_section("Analysis and Outlook", find_section(sections, "outlook"), styles))

    # 6 ------------------------------------------------------ Recommendation
    story.extend(_recommendation_flowables(result, sections, styles))
    confidence_section = _confidence_flowables(result, styles)
    if confidence_section:
        story.extend(confidence_section)
    else:
        story.extend(_section("Confidence Score", find_section(sections, "confidence"), styles))

    # 7 -------------------------------------------------------- Methodology
    methodology = find_section(sections, "methodology") or _methodology_text(result)
    story.extend(_section("Methodology & Limitations", methodology, styles))

    # 8 ---------------------------------------------------------- Disclaimer
    disclaimer = find_section(sections, "disclaimer") or DISCLAIMER_TEXT
    story.append(Paragraph("Disclaimer", styles["h1"]))
    story.append(_callout(inline_markup(disclaimer), styles, colors.HexColor("#C0392B")))

    # Anything the model added beyond the contract (e.g. Company Snapshot,
    # Risk Assessment) is preserved in an appendix rather than dropped.
    extra_headings = {alias for key in SECTION_ALIASES.values() for alias in key}
    unknown = {
        heading: body
        for heading, body in sections.items()
        if heading not in extra_headings and heading not in {"synthetic"}
    }
    if unknown:
        story.append(PageBreak())
        story.append(Paragraph("Additional analysis", styles["h1"]))
        for heading, body in unknown.items():
            story.append(Paragraph(inline_markup(heading.title()), styles["h2"]))
            story.extend(markdown_to_flowables(body, styles))

    if include_conversation:
        story.extend(_conversation_flowables(conversation, styles))
    return story


def _methodology_text(result: Any) -> str:
    """Deterministic methodology note derived from the run's own KPIs."""
    kpis = getattr(result, "kpis", {}) or {}
    ticker = getattr(result, "ticker", "the stock")
    source = kpis.get("source", "yfinance")
    days = kpis.get("trading_days", "n/a")
    risk_free = kpis.get("risk_free_rate_pct", 4.0)
    mode = (
        "live Gemini agents"
        if getattr(result, "llm_used", False)
        else "the deterministic offline crew"
    )
    return (
        f"Data for {ticker} was retrieved from {source} for "
        f"{getattr(result, 'start', '')} to {getattr(result, 'end', '')} "
        f"({days} trading days) and analysed by {mode}. KPIs are computed from daily "
        f"closes: ROI, annualised return, annualised volatility (sqrt(252) scaling), Sharpe "
        f"ratio with a {risk_free}% risk-free rate, maximum drawdown, Wilder RSI(14), "
        "SMA 20/50/200 and 20/60-day momentum. Sentiment is scored per headline with a "
        "finance-tuned lexicon model and aggregated to a -1 to +1 score. Charts are rendered "
        "with Plotly and rasterised for this document.\n\n"
        "Limitations: no fundamental filings analysis, no options-implied volatility, no "
        "intraday data, and news coverage may be incomplete. Sentiment is headline-based and "
        "backward-looking. This is a systematic, rule-assisted view, not a substitute for "
        "due diligence."
    )


def generate_pdf_report(
    result: Any,
    conversation: Any = None,
    output_path: str | Path | None = None,
    *,
    include_conversation: bool = True,
) -> Path:
    """Generate a PDF for one analysis run.

    Args:
        result: An :class:`agents.orchestrator.AnalysisResult` (duck-typed).
        conversation: A :class:`utils.logger.ConversationLog` or a list of event dicts.
        output_path: Destination file. Defaults to
            ``reports/output/pdfs/<TICKER>_<timestamp>_<run_id>.pdf``.
        include_conversation: Append the agent conversation log appendix.

    Returns:
        Path to the written PDF.

    Raises:
        RuntimeError: If the PDF cannot be written.
    """
    settings = get_settings()
    settings.ensure_directories()

    ticker = str(getattr(result, "ticker", "report"))
    run_id = str(getattr(result, "run_id", "") or "")
    if output_path is None:
        stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        filename = f"{ticker}_{stamp}{('_' + run_id) if run_id else ''}.pdf"
        destination = settings.pdf_dir / filename
    else:
        destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)

    styles = _build_styles()
    footer = f"{ticker} · {getattr(result, 'start', '')} to {getattr(result, 'end', '')}"

    doc = BaseDocTemplate(
        str(destination),
        pagesize=A4,
        leftMargin=MARGIN,
        rightMargin=MARGIN,
        topMargin=18 * mm,
        bottomMargin=18 * mm,
        title=f"{ticker} Financial Analysis Report",
        author=settings.app_name,
        subject="Automated financial analysis",
        creator=f"{settings.app_name} v{settings.app_version}",
    )
    frame = Frame(
        doc.leftMargin,
        doc.bottomMargin,
        doc.width,
        doc.height,
        id="main",
        leftPadding=0,
        rightPadding=0,
        topPadding=0,
        bottomPadding=0,
    )
    doc.addPageTemplates(
        [
            PageTemplate(
                id="report", frames=[frame], onPage=_make_page_callbacks(settings.app_name, footer)
            )
        ]
    )

    story = build_pdf_story(result, conversation, styles, include_conversation)
    try:
        doc.build(story)
    except Exception as exc:  # noqa: BLE001
        logger.exception("Failed to build PDF for %s", ticker)
        raise RuntimeError(f"PDF generation failed: {exc}") from exc

    if not destination.exists() or destination.stat().st_size < 1000:
        raise RuntimeError("PDF generation produced an empty file")

    logger.info("PDF written to %s (%s bytes)", destination, destination.stat().st_size)
    return destination


__all__ = [
    "build_pdf_story",
    "find_section",
    "generate_pdf_report",
    "inline_markup",
    "markdown_to_flowables",
    "sanitize_text",
    "split_sections",
]
