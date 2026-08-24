"""Phase 16 — MCP tools as LangChain tools.

Run:
    uv run python examples/07_mcp/langchain_adapter.py

Teaches:
* adapting MCP-discovered tools into LangChain `StructuredTool`s,
* binding them to a model so MCP capabilities join the tool-calling loop,
* why we hand-rolled the adapter (langchain-mcp-adapters pins the old mcp SDK).

Works offline (tools invoked directly). With a provider key the model can pick
MCP tools itself.
"""

from __future__ import annotations

import asyncio

from ai_research_agent.mcp.adapters import to_langchain_tools
from ai_research_agent.mcp.client import MultiMCPClient, default_servers
from ai_research_agent.models import OfflineChatModel, get_chat_model
from ai_research_agent.utils import console


async def main() -> None:
    async with MultiMCPClient(default_servers()) as client:
        tools = to_langchain_tools(client)
        console.rule("[bold]MCP tools adapted to LangChain[/]")
        for t in tools:
            console.print(f"[cyan]{t.name}[/] — {t.description}")

        console.rule("[bold]Invoke an adapted tool directly[/]")
        calc = next(t for t in tools if t.name == "analysis__calculate")
        console.print("analysis__calculate('12000 * 0.15') =", await calc.ainvoke({"expression": "12000 * 0.15"}))

        model = get_chat_model()
        if isinstance(model, OfflineChatModel):
            console.print("\n[yellow]Offline: skipping model tool-calling (needs a real model).[/]")
            return

        console.rule("[bold]Model picks an MCP tool[/]")
        bound = model.bind_tools(tools)
        ai = bound.invoke("Search internal knowledge for enterprise SOC2 requirements.")
        for call in ai.tool_calls:
            console.print(f"→ model chose [cyan]{call['name']}[/]({call['args']})")


if __name__ == "__main__":
    asyncio.run(main())
