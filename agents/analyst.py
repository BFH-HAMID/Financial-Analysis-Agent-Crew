"""Data Analyst agent: quantitative KPIs, risk metrics and charts."""

from __future__ import annotations

from typing import Any

from prompts.prompts import ANALYST
from tools.chart_tools import get_market_chart
from tools.finance_tools import get_financial_kpis, get_stock_snapshot
from utils.logger import get_logger

logger = get_logger(__name__)

NAME = "Data Analyst"
ROLE = "Computes valuation, risk and momentum KPIs and builds charts"


def build_analyst(model: Any) -> Any:
    """Create the Data Analyst agent.

    Args:
        model: An agno model instance.

    Returns:
        A configured ``agno.agent.Agent``.
    """
    from agno.agent import Agent

    agent = Agent(
        name=NAME,
        role=ROLE,
        model=model,
        tools=[get_stock_snapshot, get_financial_kpis, get_market_chart],
        instructions=[ANALYST],
        description=(
            "Quantitative analyst who interprets valuation, volatility, momentum and "
            "drawdown metrics, and flags statistical caveats in the data."
        ),
        markdown=True,
        add_datetime_to_context=True,
        debug_mode=False,
    )
    logger.debug("Built %s agent with 3 finance/chart tools", NAME)
    return agent


__all__ = ["NAME", "ROLE", "build_analyst"]
