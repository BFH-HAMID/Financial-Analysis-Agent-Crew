"""Shared pytest fixtures.

The environment is configured *before* the application modules import
``config.settings`` so the whole test-suite runs offline, deterministically and
without any API keys.
"""

from __future__ import annotations

import os
import sys
from datetime import date, timedelta
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# Deterministic, offline defaults for the test session.
os.environ.update(
    {
        "FORCE_DEMO_DATA": "1",
        "ALLOW_DEMO_DATA": "1",
        "LOG_LEVEL": "WARNING",
        "TELEMETRY_DISABLED": "1",
        "AGNO_TELEMETRY": "false",
        "MAX_REVISION_ROUNDS": "2",
        "MOCK_EXTERNAL_CALLS": "1",
    }
)
os.environ.pop("GOOGLE_API_KEY", None)


@pytest.fixture(scope="session")
def project_root() -> Path:
    """Absolute path to the repository root."""
    return ROOT


@pytest.fixture(scope="session")
def date_range() -> tuple[date, date]:
    """A stable 6-month window used across tests."""
    end = date(2024, 6, 28)
    start = end - timedelta(days=180)
    return start, end


@pytest.fixture(scope="session")
def stock(date_range):
    """Synthetic (demo) stock data for AAPL."""
    from tools.finance_tools import fetch_stock_data

    start, end = date_range
    return fetch_stock_data("AAPL", start, end)


@pytest.fixture(scope="session")
def kpis(stock):
    """KPIs computed from the demo stock data."""
    from tools.finance_tools import compute_kpis

    return compute_kpis(stock)


@pytest.fixture(scope="session")
def news():
    """Synthetic headlines for AAPL."""
    from tools.demo_data import demo_news

    return demo_news("AAPL", "Apple Inc.", 6)


@pytest.fixture(scope="session")
def sentiment(news):
    """Sentiment analysis of the synthetic headlines."""
    from tools.sentiment import analyze_texts

    return analyze_texts(news)


@pytest.fixture(scope="session")
def analysis_result(date_range):
    """A complete offline analysis run (report, charts, PDF)."""
    from agents.orchestrator import run_analysis

    start, end = date_range
    return run_analysis("AAPL", start, end, use_llm=False)
