"""Phase 20 — persistence, interrupts, and human-in-the-loop."""

from __future__ import annotations

from langgraph.types import Command

from ai_research_agent.graph.hitl import build_hitl_graph, route_review


def _cfg(thread: str) -> dict:
    return {"configurable": {"thread_id": thread}}


def test_graph_interrupts_for_review() -> None:
    graph = build_hitl_graph()
    cfg = _cfg("a")
    out = graph.invoke({"question": "Enter the market?"}, cfg)
    assert "__interrupt__" in out  # paused
    assert graph.get_state(cfg).next == ("human_review",)


def test_resume_with_approval_finishes() -> None:
    graph = build_hitl_graph()
    cfg = _cfg("b")
    graph.invoke({"question": "Enter the market?"}, cfg)
    final = graph.invoke(Command(resume={"approved": True, "feedback": ""}), cfg)
    assert final["approved"] is True
    assert final["answer"]
    assert graph.get_state(cfg).next == ()  # done


def test_rejection_loops_then_stops_at_cap() -> None:
    graph = build_hitl_graph()
    cfg = _cfg("c")
    graph.invoke({"question": "Enter the market?"}, cfg)
    # Reject repeatedly; the revision cap must eventually terminate the graph.
    graph.invoke(Command(resume={"approved": False, "feedback": "add sources"}), cfg)
    for _ in range(5):
        if graph.get_state(cfg).next == ():
            break
        graph.invoke(Command(resume={"approved": False, "feedback": "more"}), cfg)
    assert graph.get_state(cfg).next == ()  # terminated by the cap


def test_router_logic() -> None:
    assert route_review({"approved": True}) == "done"
    assert route_review({"approved": False, "iterations": 0}) == "revise"
    assert route_review({"approved": False, "iterations": 3}) == "done"


def test_state_persists_under_thread_id() -> None:
    graph = build_hitl_graph()
    cfg = _cfg("d")
    graph.invoke({"question": "Persisted question?"}, cfg)
    # The checkpoint remembers the question across separate calls.
    assert graph.get_state(cfg).values["question"] == "Persisted question?"
