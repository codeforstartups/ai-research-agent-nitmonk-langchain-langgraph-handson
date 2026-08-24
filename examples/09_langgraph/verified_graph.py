"""Phase 23 — fact-check + critic verification cycle.

Run:
    uv run python examples/09_langgraph/verified_graph.py

Teaches:
* a grounding fact-check that emits a typed Verdict,
* a conditional cycle that loops back when the answer isn't grounded (capped),
* a critic pass that sharpens the grounded answer.
"""

from __future__ import annotations

from ai_research_agent.graph.verification import build_verified_graph
from ai_research_agent.utils import console

QUESTION = "Should an enterprise startup enter the AI coding-agent market?"


def main() -> None:
    graph = build_verified_graph()
    console.rule("[bold]Graph[/]")
    console.print(graph.get_graph().draw_ascii())

    console.rule("[bold]Node order[/]")
    for update in graph.stream({"question": QUESTION}):
        for node in update:
            console.print(f"• {node}")

    state = graph.invoke({"question": QUESTION})
    console.rule("[bold]Verdict[/]")
    console.print(f"passed: {state['verdict'].passed}  issues: {state['verdict'].issues}")
    console.rule("[bold]Final answer[/]")
    console.print(state["answer"][:400])


if __name__ == "__main__":
    main()
