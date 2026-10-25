"""Report Writer agent: composes the final Markdown report."""

from __future__ import annotations

from typing import Any

from prompts.prompts import WRITER
from utils.logger import get_logger

logger = get_logger(__name__)

NAME = "Report Writer"
ROLE = "Composes the final client-ready report with recommendation and confidence score"


def build_writer(model: Any) -> Any:
    """Create the Report Writer agent (no tools: it synthesises upstream output).

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
        tools=[],
        instructions=[WRITER],
        description=(
            "Financial writer who turns research and analytics into a structured, "
            "balanced report with an explicit recommendation and confidence score."
        ),
        markdown=True,
        add_datetime_to_context=True,
        debug_mode=False,
    )
    logger.debug("Built %s agent (no tools)", NAME)
    return agent


__all__ = ["NAME", "ROLE", "build_writer"]
