"""Phase 14 — the Knowledge MCP server (internal RAG over MCP).

Run as a stdio server:
    uv run python examples/07_mcp/knowledge_server.py

Inspect it:
    npx @modelcontextprotocol/inspector uv run python examples/07_mcp/knowledge_server.py

Teaches:
* exposing internal RAG as an MCP server, so it's discovered and called exactly
  like the web tools — one uniform interface over two context sources.
"""

from __future__ import annotations

import asyncio

from ai_research_agent.mcp.servers import build_knowledge_server

server = build_knowledge_server()


if __name__ == "__main__":
    asyncio.run(server.run_stdio_async())
