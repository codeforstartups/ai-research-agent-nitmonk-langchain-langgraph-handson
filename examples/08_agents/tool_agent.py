"""Phase 17 — a tool-calling agent over the MCP toolset.

Run:
    uv run python examples/08_agents/tool_agent.py

Teaches:
* the explicit agent loop (model → tools → model → answer),
* the agent drawing tools from all MCP servers via the adapter,
* iteration caps as a safety rail.

Needs a provider key to actually call tools; offline it returns the stub answer
and no steps (which is itself instructive — no key, no autonomy).
"""

from __future__ import annotations

import asyncio

from ai_research_agent.agents import run_tool_agent
from ai_research_agent.mcp.adapters import to_langchain_tools
from ai_research_agent.mcp.client import MultiMCPClient, default_servers
from ai_research_agent.utils import console

QUESTION = "What internal requirements must an enterprise AI coding agent meet? Search internal knowledge."


async def main() -> None:
    async with MultiMCPClient(default_servers()) as client:
        tools = to_langchain_tools(client)
        result = await run_tool_agent(QUESTION, tools)

    console.rule("[bold]Tool steps[/]")
    for step in result.steps:
        console.print(f"→ [cyan]{step.tool}[/]({step.args}) → {step.output[:80]}…")
    console.rule(f"[bold]Answer[/] (in {result.iterations} iterations)")
    console.print(result.answer)


if __name__ == "__main__":
    asyncio.run(main())
