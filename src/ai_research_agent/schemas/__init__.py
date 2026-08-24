"""Typed I/O schemas.

Anything that crosses a boundary — model outputs, tool arguments, MCP payloads —
gets a Pydantic schema so it is validated, self-documenting, and safe to pass
between phases.
"""

from ai_research_agent.schemas.research import ResearchPlan, SubQuestion

__all__ = ["ResearchPlan", "SubQuestion"]
