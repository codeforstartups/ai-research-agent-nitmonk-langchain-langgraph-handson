"""Phase 13 — the Web Research MCP server (Thordata-backed).

Run as a stdio server:
    uv run python examples/07_mcp/web_research_server.py

Inspect it:
    npx @modelcontextprotocol/inspector uv run python examples/07_mcp/web_research_server.py

Teaches:
* wrapping a real external API (Thordata) behind an MCP server,
* the untrusted-content boundary applied to everything from the web,
* resources that expose *application state* (the sources gathered so far).

Runs with no Thordata key using deterministic offline fixtures; set
THORDATA_API_KEY in .env to hit the live SERP API.
"""

from __future__ import annotations

import asyncio

from ai_research_agent.mcp.servers.web_research import build_web_research_server

server = build_web_research_server()


if __name__ == "__main__":
    asyncio.run(server.run_stdio_async())
