"""Persistence, checkpoints, interrupts, and human-in-the-loop.

Two capabilities LangGraph gives us that a plain loop can't, cheaply:

* **Persistence / checkpoints** — attach a *checkpointer* and every step's state
  is saved under a ``thread_id``. The run can be inspected, resumed after a
  crash, or continued days later.
* **Interrupts / human-in-the-loop** — a node calls ``interrupt(payload)``; the
  graph *pauses* and returns control. A human reviews, then the caller resumes
  with ``Command(resume=decision)`` and execution continues from exactly where
  it stopped (thanks to the checkpoint).

Flow:

    START → plan → research → draft → human_review ─approved→ END
                                        │
                                        └─rejected→ revise → human_review (loop)
"""

from __future__ import annotations

from pathlib import Path

from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph
from langgraph.types import interrupt

from ai_research_agent.graph.nodes import plan_node, research_node, synthesize_node
from ai_research_agent.graph.state import ResearchState

MAX_REVISIONS = 3

# Checkpointed state includes our ResearchPlan (a Pydantic model). LangGraph
# will not deserialise arbitrary modules by default (a security measure); we
# explicitly allow our own schema module.
_ALLOWED_MSGPACK_MODULES = [("ai_research_agent.schemas.research", "ResearchPlan")]


def _serializer():
    from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer

    return JsonPlusSerializer(allowed_msgpack_modules=_ALLOWED_MSGPACK_MODULES)


def draft_node(state: ResearchState) -> ResearchState:
    """Produce a draft answer (same as synthesize, but subject to review)."""
    return synthesize_node(state)


def human_review_node(state: ResearchState) -> ResearchState:
    """Pause for human review of the draft.

    ``interrupt`` suspends the graph and surfaces the payload to the caller. The
    value returned here is whatever the human passes via ``Command(resume=...)``.
    """
    decision = interrupt(
        {
            "question": state["question"],
            "draft": state.get("answer", ""),
            "instructions": "Reply with {'approved': bool, 'feedback': str}.",
        }
    )
    decision = decision or {}
    return {
        "approved": bool(decision.get("approved", False)),
        "human_feedback": str(decision.get("feedback", "")),
    }


def revise_node(state: ResearchState) -> ResearchState:
    """Incorporate human feedback and re-draft."""
    feedback = state.get("human_feedback", "")
    augmented = dict(state)
    if feedback:
        augmented["evidence"] = [*state.get("evidence", []), f"[human-feedback] {feedback}"]
    update = synthesize_node(augmented)  # re-synthesize with feedback in evidence
    update["iterations"] = state.get("iterations", 0) + 1
    return {**update, "evidence": [f"[human-feedback] {feedback}"] if feedback else []}


def route_review(state: ResearchState) -> str:
    """Approved → done; rejected → revise (bounded by MAX_REVISIONS)."""
    if state.get("approved") or state.get("iterations", 0) >= MAX_REVISIONS:
        return "done"
    return "revise"


def build_hitl_graph(*, checkpointer=None, sqlite_path: str | Path | None = None) -> CompiledStateGraph:
    """Build the human-in-the-loop graph with a checkpointer.

    A checkpointer is **required** for interrupts. Pass one, or a
    ``sqlite_path`` for durable on-disk persistence; defaults to in-memory.
    """
    if checkpointer is None:
        if sqlite_path is not None:
            import sqlite3

            from langgraph.checkpoint.sqlite import SqliteSaver

            conn = sqlite3.connect(str(sqlite_path), check_same_thread=False)
            checkpointer = SqliteSaver(conn, serde=_serializer())
        else:
            checkpointer = InMemorySaver(serde=_serializer())

    graph = StateGraph(ResearchState)
    graph.add_node("plan", plan_node)
    graph.add_node("research", research_node)
    graph.add_node("draft", draft_node)
    graph.add_node("human_review", human_review_node)
    graph.add_node("revise", revise_node)

    graph.add_edge(START, "plan")
    graph.add_edge("plan", "research")
    graph.add_edge("research", "draft")
    graph.add_edge("draft", "human_review")
    graph.add_conditional_edges("human_review", route_review, {"revise": "revise", "done": END})
    graph.add_edge("revise", "human_review")

    return graph.compile(checkpointer=checkpointer)
