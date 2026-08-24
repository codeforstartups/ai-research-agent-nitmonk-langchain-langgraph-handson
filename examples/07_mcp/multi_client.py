"""Phase 15 — one client, many MCP servers.

Run:
    uv run python examples/07_mcp/multi_client.py

Teaches:
* connecting to several MCP servers and discovering their tools at runtime,
* namespacing (`server__tool`) so names never collide,
* routing a call to whichever server owns the tool.
"""

from __future__ import annotations

import asyncio

from ai_research_agent.mcp.client import MultiMCPClient, default_servers
from ai_research_agent.utils import console


async def main() -> None:
    async with MultiMCPClient(default_servers()) as client:
        console.rule("[bold]Discovered tools[/]")
        for tool in client.list_tools():
            console.print(f"[cyan]{tool.qualified_name}[/] — {tool.description}")

        console.rule("[bold]Call one tool from each server[/]")
        calc = await client.call_tool("analysis__calculate", {"expression": "12000 * 0.15"})
        console.print(f"analysis__calculate('12000 * 0.15') = {calc}")

        web = await client.call_tool("web_research__search_web", {"query": "AI coding agents", "count": 2})
        console.print(f"web_research__search_web → {web.splitlines()[0]} …")

        kb = await client.call_tool("knowledge__list_sources", {})
        console.print(f"knowledge__list_sources →\n{kb}")


if __name__ == "__main__":
    asyncio.run(main())
