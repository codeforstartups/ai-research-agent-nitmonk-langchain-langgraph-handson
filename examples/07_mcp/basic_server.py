"""Phase 12 — a first MCP server (run it, inspect it).

Run as a stdio server (what an MCP host launches):
    uv run python examples/07_mcp/basic_server.py

Explore it interactively with the MCP Inspector:
    npx @modelcontextprotocol/inspector uv run python examples/07_mcp/basic_server.py

Teaches:
* an MCP server exposes tools + resources + prompts over a transport (stdio),
* the host/client launches the server and discovers its capabilities,
* the Inspector is your 'Postman for MCP' during development.
"""

from __future__ import annotations

import asyncio

from ai_research_agent.mcp.servers import build_analysis_server

# The server object the Inspector / host connects to.
server = build_analysis_server()


if __name__ == "__main__":
    # Serve over stdio: read requests on stdin, write responses on stdout.
    asyncio.run(server.run_stdio_async())
