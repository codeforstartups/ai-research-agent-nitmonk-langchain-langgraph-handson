"""Phase 11 — tools and tool calling.

Run:
    uv run python examples/06_tools/tool_calling.py

Teaches:
* defining tools with `@tool` (signature + docstring → schema),
* `model.bind_tools(...)` so the model can *request* tool calls,
* the tool-calling loop: model asks → we execute → we feed results back.

Tool calling needs a real provider (the offline stub can't emit tool calls), so
without a key we just demonstrate invoking the tools directly.
"""

from __future__ import annotations

from langchain_core.messages import HumanMessage, ToolMessage

from ai_research_agent.models import OfflineChatModel, get_chat_model
from ai_research_agent.tools import ALL_TOOLS, calculate, search_knowledge
from ai_research_agent.utils import console

TOOLS_BY_NAME = {t.name: t for t in ALL_TOOLS}


def main() -> None:
    console.rule("[bold]Tools invoked directly[/]")
    console.print("calculate('12000 * 0.15') =", calculate.invoke({"expression": "12000 * 0.15"}))
    console.print(
        "search_knowledge(...) →",
        search_knowledge.invoke({"query": "enterprise requirements"})[:120],
        "…",
    )

    model = get_chat_model()
    if isinstance(model, OfflineChatModel):
        console.print("\n[yellow]Offline: skipping the tool-calling loop (needs a real model).[/]")
        return

    console.rule("[bold]Tool-calling loop[/]")
    bound = model.bind_tools(ALL_TOOLS)
    messages = [HumanMessage("What is 15% of a 12000 (in $M) market? Use the calculator.")]
    ai = bound.invoke(messages)
    messages.append(ai)

    for call in ai.tool_calls:
        console.print(f"→ model called [cyan]{call['name']}[/]({call['args']})")
        output = TOOLS_BY_NAME[call["name"]].invoke(call["args"])
        messages.append(ToolMessage(content=str(output), tool_call_id=call["id"]))

    final = bound.invoke(messages)
    console.print(f"\n[bold]Final:[/] {final.text}")


if __name__ == "__main__":
    main()
