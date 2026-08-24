"""Phase 22 — supervisor coordinating specialist subagents.

Run:
    uv run python examples/09_langgraph/supervisor.py

Teaches:
* specialist subagents, each with a narrow remit and a typed Finding output,
* a supervisor that dispatches them in parallel and an aggregator that merges,
* why focused agents beat one generalist prompt.
"""

from __future__ import annotations

from ai_research_agent.graph.supervisor import build_supervisor_graph
from ai_research_agent.utils import console

QUESTION = "Should an enterprise startup enter the AI coding-agent market?"


def main() -> None:
    graph = build_supervisor_graph()
    console.rule("[bold]Graph[/]")
    console.print(graph.get_graph().draw_ascii())

    state = graph.invoke({"question": QUESTION})
    console.rule("[bold]Specialist findings[/]")
    for f in state["findings"]:
        console.print(f"[cyan]{f.facet}[/] — {len(f.sources)} sources")
    console.rule("[bold]Aggregated answer[/]")
    console.print(state["answer"][:400])


if __name__ == "__main__":
    main()
