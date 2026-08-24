"""MCP client layer: connect to and aggregate multiple MCP servers."""

from ai_research_agent.mcp.client.multi_client import (
    DiscoveredTool,
    MultiMCPClient,
    default_servers,
)

__all__ = ["DiscoveredTool", "MultiMCPClient", "default_servers"]
