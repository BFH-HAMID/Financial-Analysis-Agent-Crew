#!/usr/bin/env python3
"""Headless report generator for scheduled runs (cron / GitHub Actions / Docker).

Examples
--------
Generate reports for a watchlist into ``reports/output``::

    python scripts/daily_report.py --tickers AAPL,MSFT,NVDA --days 180

Offline demo (no keys, synthetic data, clearly labelled)::

    FORCE_DEMO_DATA=1 python scripts/daily_report.py --tickers AAPL --no-llm

Send a Telegram summary (requires ``TELEGRAM_BOT_TOKEN`` + ``TELEGRAM_CHAT_ID``)::

    python scripts/daily_report.py --tickers AAPL --telegram

Exit code is 0 when every ticker produced a report, 1 otherwise, so CI can
surface failures.
"""

from __future__ import annotations

import argparse
import json
import os
import smtplib
import sys
from datetime import date, timedelta
from email.message import EmailMessage
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agents.orchestrator import AnalysisResult, run_analysis  # noqa: E402
from config.settings import get_settings  # noqa: E402
from utils.formatting import fmt_confidence, fmt_currency, fmt_pct  # noqa: E402
from utils.logger import get_logger  # noqa: E402

logger = get_logger(__name__)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    settings = get_settings()
    parser = argparse.ArgumentParser(
        description="Generate financial analysis reports without the UI.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--tickers",
        default=os.getenv("DAILY_TICKERS", "AAPL"),
        help="Comma-separated tickers, e.g. AAPL,MSFT,NVDA",
    )
    parser.add_argument("--days", type=int, default=180, help="Look-back window in days")
    parser.add_argument("--start", help="Explicit start date (YYYY-MM-DD); overrides --days")
    parser.add_argument("--end", help="Explicit end date (YYYY-MM-DD); defaults to today")
    parser.add_argument("--no-llm", action="store_true", help="Force the offline crew")
    parser.add_argument("--no-pdf", action="store_true", help="Skip PDF generation (Markdown only)")
    parser.add_argument("--telegram", action="store_true", help="Send a Telegram summary")
    parser.add_argument("--email", action="store_true", help="Email the summary and PDFs")
    parser.add_argument(
        "--summary-json",
        default=str(settings.reports_dir / "daily_summary.json"),
        help="Where to write the machine-readable run summary",
    )
    parser.add_argument(
        "--fail-on-empty", action="store_true", help="Exit 1 if no ticker produced a report"
    )
    return parser.parse_args(argv)


def resolve_dates(args: argparse.Namespace) -> tuple[date, date]:
    end = date.fromisoformat(args.end) if args.end else date.today()
    start = (
        date.fromisoformat(args.start) if args.start else end - timedelta(days=max(args.days, 7))
    )
    return start, end


def summarise(result: AnalysisResult) -> str:
    """One-line, Telegram-friendly summary of a run."""
    kpis = result.kpis
    return (
        f"{result.ticker} ({result.company_name[:28]}) | "
        f"{fmt_currency(kpis.get('current_price'))} | ROI {fmt_pct(kpis.get('period_roi_pct'))} | "
        f"sentiment {result.sentiment.get('label')} | {result.recommendation} | "
        f"confidence {fmt_confidence(result.confidence)}"
        + (" | DEMO DATA" if result.is_demo else "")
    )


def send_telegram(text: str) -> bool:
    """Send a message via the Telegram Bot API. Returns True on success."""
    token = os.getenv("TELEGRAM_BOT_TOKEN")
    chat_id = os.getenv("TELEGRAM_CHAT_ID")
    if not token or not chat_id:
        logger.warning("Telegram not configured (TELEGRAM_BOT_TOKEN / TELEGRAM_CHAT_ID)")
        return False
    try:
        import requests

        response = requests.post(
            f"https://api.telegram.org/bot{token}/sendMessage",
            json={"chat_id": chat_id, "text": text[:4000], "disable_web_page_preview": True},
            timeout=20,
        )
        response.raise_for_status()
        logger.info("Telegram summary sent")
        return True
    except Exception as exc:  # noqa: BLE001
        logger.error("Telegram delivery failed: %s", exc)
        return False


def send_email(subject: str, body: str, attachments: list[str]) -> bool:
    """Email the summary with PDF attachments. Returns True on success."""
    host = os.getenv("SMTP_HOST")
    sender = os.getenv("EMAIL_FROM")
    recipient = os.getenv("EMAIL_TO")
    if not (host and sender and recipient):
        logger.warning("Email not configured (SMTP_HOST / EMAIL_FROM / EMAIL_TO)")
        return False

    message = EmailMessage()
    message["Subject"] = subject
    message["From"] = sender
    message["To"] = recipient
    message.set_content(body)

    for path in attachments:
        file_path = Path(path)
        if not file_path.exists():
            continue
        message.add_attachment(
            file_path.read_bytes(),
            maintype="application",
            subtype="pdf",
            filename=file_path.name,
        )

    try:
        port = int(os.getenv("SMTP_PORT", "587"))
        with smtplib.SMTP(host, port, timeout=30) as client:
            client.starttls()
            user = os.getenv("SMTP_USER")
            password = os.getenv("SMTP_PASSWORD")
            if user and password:
                client.login(user, password)
            client.send_message(message)
        logger.info("Email sent to %s", recipient)
        return True
    except Exception as exc:  # noqa: BLE001
        logger.error("Email delivery failed: %s", exc)
        return False


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    start, end = resolve_dates(args)
    tickers = [ticker.strip().upper() for ticker in args.tickers.split(",") if ticker.strip()]
    if not tickers:
        logger.error("No tickers supplied")
        return 2

    logger.info(
        "Running scheduled analysis for %s (%s → %s, llm=%s)",
        ", ".join(tickers),
        start,
        end,
        not args.no_llm,
    )

    results: list[AnalysisResult] = []
    failures: dict[str, str] = {}

    for ticker in tickers:
        try:
            result = run_analysis(
                ticker,
                start,
                end,
                use_llm=not args.no_llm,
                generate_pdf=not args.no_pdf,
            )
            results.append(result)
            logger.info("✅ %s", summarise(result))
        except Exception as exc:  # noqa: BLE001 - one ticker must not stop the batch
            failures[ticker] = f"{type(exc).__name__}: {exc}"
            logger.error("❌ %s failed: %s", ticker, exc)

    lines = [f"📈 Financial Analysis Agent Crew — {end.isoformat()}", ""]
    lines += [f"• {summarise(result)}" for result in results]
    lines += [f"• {ticker} — FAILED ({error})" for ticker, error in failures.items()]
    if any(result.is_demo for result in results):
        lines += ["", "⚠️ Some figures are SYNTHETIC DEMO DATA."]
    summary_text = "\n".join(lines)
    print(summary_text)

    summary_path = Path(args.summary_json)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(
        json.dumps(
            {
                "generated_at": date.today().isoformat(),
                "start": start.isoformat(),
                "end": end.isoformat(),
                "runs": [
                    {
                        "ticker": result.ticker,
                        "recommendation": result.recommendation,
                        "confidence": result.confidence,
                        "roi_pct": result.kpis.get("period_roi_pct"),
                        "sentiment": result.sentiment.get("label"),
                        "is_demo": result.is_demo,
                        "pdf": result.pdf_path,
                        "report": result.report_path,
                    }
                    for result in results
                ],
                "failures": failures,
            },
            indent=2,
            default=str,
        ),
        encoding="utf-8",
    )
    logger.info("Run summary written to %s", summary_path)

    if args.telegram:
        send_telegram(summary_text)
    if args.email and results:
        send_email(
            subject=f"Daily financial analysis — {end.isoformat()}",
            body=summary_text + "\n\nReports are attached.",
            attachments=[r.pdf_path for r in results if r.pdf_path],
        )

    if args.fail_on_empty and not results:
        return 1
    return 0 if results else 1


if __name__ == "__main__":
    sys.exit(main())
