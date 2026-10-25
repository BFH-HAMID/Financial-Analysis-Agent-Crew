"""Tests for market-data retrieval and KPI maths."""

from __future__ import annotations

import json
import math

import numpy as np
import pandas as pd
import pytest

from tools.demo_data import demo_price_history, demo_profile
from tools.finance_tools import (
    FinanceDataError,
    StockData,
    annualized_volatility,
    build_kpi_table,
    compute_kpis,
    compute_max_drawdown,
    compute_rsi,
    fetch_stock_data,
    get_financial_kpis,
    get_stock_snapshot,
    sharpe_ratio,
)


class TestRsi:
    def test_monotonic_rise_is_overbought(self):
        close = pd.Series(np.linspace(100, 200, 40))
        assert compute_rsi(close) == pytest.approx(100.0)

    def test_monotonic_fall_is_oversold(self):
        close = pd.Series(np.linspace(200, 100, 40))
        assert compute_rsi(close) == pytest.approx(0.0)

    def test_constant_series_is_neutral(self):
        assert compute_rsi(pd.Series([100.0] * 30)) == pytest.approx(50.0)

    def test_insufficient_history_returns_none(self):
        assert compute_rsi(pd.Series([100.0, 101.0, 102.0])) is None

    def test_known_reference_series(self):
        """Wilder's RSI on a textbook up/down series stays in range and is deterministic."""
        values = [
            44.34,
            44.09,
            44.15,
            43.61,
            44.33,
            44.83,
            45.10,
            45.42,
            45.84,
            46.08,
            45.89,
            46.03,
            45.61,
            46.28,
            46.28,
            46.00,
            46.03,
            46.41,
            46.22,
            45.64,
        ]
        rsi = compute_rsi(pd.Series(values))
        assert rsi is not None
        assert 0 <= rsi <= 100
        assert rsi == pytest.approx(compute_rsi(pd.Series(values)))


class TestRiskMetrics:
    def test_max_drawdown_known_values(self):
        close = pd.Series([100, 120, 90, 110, 60, 100])
        assert compute_max_drawdown(close) == pytest.approx(50.0)

    def test_max_drawdown_needs_two_points(self):
        assert compute_max_drawdown(pd.Series([100.0])) is None

    def test_volatility_of_flat_series_is_zero(self):
        assert annualized_volatility(pd.Series([100.0] * 30)) == pytest.approx(0.0)

    def test_volatility_scales_with_dispersion(self):
        quiet = pd.Series(100 + np.sin(np.arange(60)) * 0.1)
        wild = pd.Series(100 + np.sin(np.arange(60)) * 10)
        assert annualized_volatility(wild) > annualized_volatility(quiet)

    def test_sharpe_none_for_flat_series(self):
        assert sharpe_ratio(pd.Series([100.0] * 30)) is None


class TestStockData:
    def test_demo_profile_is_deterministic(self):
        assert demo_profile("AAPL") == demo_profile("AAPL")
        assert demo_profile("AAPL").ticker == "AAPL"

    def test_demo_history_is_reproducible_and_ordered(self):
        from datetime import date

        first = demo_price_history("TSLA", date(2024, 1, 1), date(2024, 3, 1))
        second = demo_price_history("TSLA", date(2024, 1, 1), date(2024, 3, 1))
        pd.testing.assert_frame_equal(first, second)
        assert first.index.is_monotonic_increasing
        assert (first["High"] >= first["Low"]).all()
        assert (first["Volume"] > 0).all()

    def test_fetch_stock_data_demo(self, date_range, stock):
        start, end = date_range
        assert isinstance(stock, StockData)
        assert stock.is_demo is True
        assert stock.has_data
        assert stock.ticker == "AAPL"
        assert stock.latest_price and stock.latest_price > 0
        assert stock.start == start and stock.end == end

    def test_fetch_stock_data_raises_without_demo(self, date_range, monkeypatch):
        start, end = date_range

        def boom(*args, **kwargs):
            raise RuntimeError("provider down")

        monkeypatch.setattr("tools.finance_tools._fetch_yfinance", boom)
        with pytest.raises(FinanceDataError):
            fetch_stock_data("AAPL", start, end, allow_demo=False, force_demo=False)


class TestKpis:
    def test_required_keys_present(self, kpis):
        expected = {
            "ticker",
            "current_price",
            "period_roi_pct",
            "annualized_return_pct",
            "volatility_annual_pct",
            "sharpe_ratio",
            "max_drawdown_pct",
            "rsi_14",
            "sma_20",
            "sma_50",
            "momentum_20d_pct",
            "market_cap",
            "pe_ratio",
            "is_demo",
            "data_quality",
        }
        assert expected.issubset(kpis.keys())

    def test_values_are_finite(self, kpis):
        for key, value in kpis.items():
            if isinstance(value, float):
                assert not math.isnan(value), f"{key} is NaN"
                assert not math.isinf(value), f"{key} is infinite"

    def test_rsi_in_range(self, kpis):
        assert 0 <= kpis["rsi_14"] <= 100

    def test_demo_data_is_flagged(self, kpis):
        assert kpis["is_demo"] is True
        assert any("SYNTHETIC" in note.upper() for note in kpis["data_quality"])

    def test_kpi_table_is_human_readable(self, kpis):
        rows = build_kpi_table(kpis)
        assert len(rows) >= 15
        assert all(set(row) == {"metric", "value", "note"} for row in rows)
        assert any(row["metric"] == "P/E (trailing)" for row in rows)

    def test_missing_volume_does_not_crash(self, stock):
        history = stock.history.drop(columns=["Volume"])
        trimmed = StockData(
            ticker=stock.ticker,
            company_name=stock.company_name,
            start=stock.start,
            end=stock.end,
            history=history,
            info=stock.info,
            is_demo=True,
        )
        result = compute_kpis(trimmed)
        assert result["ticker"] == "AAPL"


class TestToolWrappers:
    def test_snapshot_returns_json(self, date_range):
        start, end = date_range
        payload = json.loads(get_stock_snapshot("AAPL", start.isoformat(), end.isoformat()))
        assert payload["ticker"] == "AAPL"
        assert payload["rows"] > 0
        assert payload["recent_closes"]

    def test_kpi_tool_returns_json(self, date_range):
        start, end = date_range
        payload = json.loads(get_financial_kpis("MSFT", start.isoformat(), end.isoformat()))
        assert payload["ticker"] == "MSFT"
        assert "volatility_annual_pct" in payload

    def test_tool_errors_are_returned_as_json(self):
        payload = json.loads(get_financial_kpis("AAPL", "not-a-date", "2024-01-01"))
        assert "error" in payload
