"""Phase 21 — LangGraph nodes calling MCP capabilities.

Run:
    uv run python examples/09_langgraph/mcp_graph.py

Teaches:
* graph nodes gather evidence through the MCP client (web + knowledge servers),
* async graph execution (`await graph.ainvoke(...)`),
* modularity: the graph doesn't know how the servers work — register a server,
  the graph can use it.
"""

from __future__ import annotations

import asyncio

from ai_research_agent.graph.mcp_graph import build_mcp_graph
from ai_research_agent.mcp.client import MultiMCPClient, default_servers
from ai_research_agent.utils import console


async def main() -> None:
    async with MultiMCPClient(default_servers()) as client:
        console.print(f"[dim]connected tools:[/] {len(client.list_tools())}")
        graph = build_mcp_graph(client)
        state = await graph.ainvoke(
            {"question": "Should an enterprise startup enter the AI coding-agent market?"}
        )

    console.rule("[bold]Evidence sources[/]")
    console.print(f"gathered {len(state['evidence'])} evidence blocks via MCP")
    console.print(f"sources: {state['sources']}")
    console.rule("[bold]Answer[/]")
    console.print(state["answer"][:400])


if __name__ == "__main__":
    asyncio.run(main())
