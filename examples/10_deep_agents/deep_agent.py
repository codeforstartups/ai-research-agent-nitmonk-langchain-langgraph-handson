"""Phase 24 — the research agent as a Deep Agent.

Run:
    uv run python examples/10_deep_agents/deep_agent.py

Teaches:
* what the Deep Agents harness gives you for free: planning (todos), a
  filesystem for context, and delegable subagents,
* the comparison with the hand-built LangGraph implementation,
* streaming the agent's messages.

Needs a provider key to actually plan/delegate/call tools; offline it returns
the stub response (no autonomy without a model).
"""

from __future__ import annotations

from ai_research_agent.deep_agent import build_research_deep_agent
from ai_research_agent.models import OfflineChatModel, get_chat_model
from ai_research_agent.utils import console

QUESTION = "Analyze the AI coding-agent market and whether an enterprise startup should enter it."


def main() -> None:
    agent = build_research_deep_agent()

    console.rule("[bold]Deep Agent structure[/]")
    console.print("nodes:", list(agent.get_graph().nodes))
    console.print(
        "subagents: market-researcher, competitor-analyst, enterprise-analyst"
    )

    if isinstance(get_chat_model(), OfflineChatModel):
        console.print("\n[yellow]Offline: the harness runs but won't plan/delegate without a model.[/]")

    console.rule("[bold]Run[/]")
    result = agent.invoke({"messages": [{"role": "user", "content": QUESTION}]})
    final = result["messages"][-1]
    console.print(final.text if hasattr(final, "text") else str(final))


if __name__ == "__main__":
    main()
