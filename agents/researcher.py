"""Market Researcher agent: news gathering and sentiment interpretation."""

from __future__ import annotations

from typing import Any

from prompts.prompts import RESEARCHER
from tools.news_tools import search_stock_news
from tools.sentiment import analyze_news_sentiment
from utils.logger import get_logger

logger = get_logger(__name__)

NAME = "Market Researcher"
ROLE = "Gathers recent news, groups themes and judges market sentiment"


def build_researcher(model: Any) -> Any:
    """Create the Market Researcher agent.

    Args:
        model: An agno model instance (usually Gemini via :func:`agents.llm.build_model`).

    Returns:
        A configured ``agno.agent.Agent``.
    """
    from agno.agent import Agent

    agent = Agent(
        name=NAME,
        role=ROLE,
        model=model,
        tools=[search_stock_news, analyze_news_sentiment],
        instructions=[RESEARCHER],
        description=(
            "Senior market researcher who reads financial news critically and "
            "quantifies sentiment with transparent, reproducible scores."
        ),
        markdown=True,
        add_datetime_to_context=True,
        debug_mode=False,
    )
    logger.debug(
        "Built %s agent with tools: %s",
        NAME,
        [t.__name__ for t in (search_stock_news, analyze_news_sentiment)],
    )
    return agent


__all__ = ["NAME", "ROLE", "build_researcher"]
