"""Phase 20 — persistence, interrupts, and human-in-the-loop.

Run:
    uv run python examples/09_langgraph/human_in_the_loop.py

Teaches:
* a checkpointer persists state under a thread_id,
* `interrupt()` pauses the graph for human review,
* `get_state()` inspects the paused run; `Command(resume=...)` continues it.

This script simulates the human by auto-approving; a real UI would collect the
decision. Runs offline (no keys).
"""

from __future__ import annotations

from langgraph.types import Command

from ai_research_agent.graph.hitl import build_hitl_graph
from ai_research_agent.utils import console


def main() -> None:
    graph = build_hitl_graph()
    config = {"configurable": {"thread_id": "demo-thread"}}

    console.rule("[bold]Run until human review[/]")
    result = graph.invoke({"question": "Should we enter the AI coding-agent market?"}, config)
    interrupts = result.get("__interrupt__", [])
    if interrupts:
        payload = interrupts[0].value
        console.print("[yellow]⏸ Paused for human review.[/]")
        console.print(f"Draft (first 120 chars): {payload['draft'][:120]}…")

    console.rule("[bold]Inspect the checkpoint[/]")
    snap = graph.get_state(config)
    console.print(f"Paused at node: {snap.next}")

    console.rule("[bold]Human approves → resume[/]")
    final = graph.invoke(Command(resume={"approved": True, "feedback": ""}), config)
    console.print(f"Approved: {final['approved']}")
    console.print(f"Graph state next (empty = done): {graph.get_state(config).next}")


if __name__ == "__main__":
    main()
