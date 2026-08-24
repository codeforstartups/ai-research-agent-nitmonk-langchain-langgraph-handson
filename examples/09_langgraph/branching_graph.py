"""Phase 19 — parallel fan-out, conditional routing, and cycles.

Run:
    uv run python examples/09_langgraph/branching_graph.py

Teaches:
* running two research branches (web + internal) in parallel and joining them,
* a conditional edge that loops back for more evidence (a cycle),
* a termination guard so the cycle always ends.
"""

from __future__ import annotations

from ai_research_agent.graph import build_branching_graph
from ai_research_agent.utils import console

QUESTION = "Should an enterprise startup enter the AI coding-agent market?"


def main() -> None:
    graph = build_branching_graph()
    console.rule("[bold]Graph structure[/]")
    console.print(graph.get_graph().draw_ascii())

    console.rule("[bold]Streaming node execution[/]")
    final = {}
    for update in graph.stream({"question": QUESTION}):
        for node, _ in update.items():
            console.print(f"• ran node: [cyan]{node}[/]")
        final.update(update)

    state = graph.invoke({"question": QUESTION})
    console.rule("[bold]Result[/]")
    console.print(
        f"loops: {state.get('iterations', 0)}  •  evidence: {len(state['evidence'])}  "
        f"•  web+internal sources: {len(state['sources'])}"
    )


if __name__ == "__main__":
    main()
