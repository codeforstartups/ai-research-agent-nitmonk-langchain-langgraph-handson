"""Adapters that expose MCP capabilities to other frameworks (LangChain)."""

from ai_research_agent.mcp.adapters.langchain import (
    mcp_tool_to_langchain,
    to_langchain_tools,
)

__all__ = ["mcp_tool_to_langchain", "to_langchain_tools"]
