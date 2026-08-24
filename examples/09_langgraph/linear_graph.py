"""Phase 18 — a first LangGraph (State → nodes → edges).

Run:
    uv run python examples/09_langgraph/linear_graph.py

Teaches:
* the typed State object threaded between nodes,
* nodes as plain functions returning partial updates,
* edges wiring START → plan → research → synthesize → END,
* ASCII visualization of the compiled graph.
"""

from __future__ import annotations

from ai_research_agent.graph import build_research_graph
from ai_research_agent.utils import console

QUESTION = "Should an enterprise startup enter the AI coding-agent market?"


def main() -> None:
    graph = build_research_graph()

    console.rule("[bold]Graph structure[/]")
    console.print(graph.get_graph().draw_ascii())

    console.rule("[bold]Run[/]")
    state = graph.invoke({"question": QUESTION})
    console.print(f"Plan: {len(state['plan'].sub_questions)} sub-questions")
    console.print(f"Evidence: {len(state['evidence'])} chunks from {len(state['sources'])} sources")
    console.rule("[bold]Answer[/]")
    console.print(state["answer"])


if __name__ == "__main__":
    main()
