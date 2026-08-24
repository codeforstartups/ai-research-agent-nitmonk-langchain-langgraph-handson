"""LangChain tools the agent can call.

These are plain LangChain tools. In the MCP phases (12+) some of this same
capability is re-exposed through MCP servers — the contrast between "a local
@tool" and "a tool discovered over a protocol" is a core teaching point.
"""

from ai_research_agent.tools.analysis import calculate, percentage_change
from ai_research_agent.tools.knowledge import search_knowledge

# The default toolset bound to the agent.
ALL_TOOLS = [search_knowledge, calculate, percentage_change]

__all__ = ["ALL_TOOLS", "calculate", "percentage_change", "search_knowledge"]
