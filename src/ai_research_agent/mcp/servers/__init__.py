"""MCP servers, organised by responsibility (not one giant server).

* ``analysis``       — deterministic computation tools (phase 12).
* ``web_research``   — live web search/scrape via Thordata (phase 13).
* ``knowledge``      — internal RAG exposed over MCP (phase 14).

Each ``build_*_server()`` returns a configured :class:`mcp.server.MCPServer`
that can be run over stdio/HTTP *or* connected in-memory for tests.
"""

from ai_research_agent.mcp.servers.analysis import build_analysis_server
from ai_research_agent.mcp.servers.web_research import build_web_research_server

__all__ = ["build_analysis_server", "build_web_research_server"]
