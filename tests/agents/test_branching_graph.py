"""Phase 19 — parallel fan-out, conditional routing, and cycles."""

from __future__ import annotations

from ai_research_agent.graph import build_branching_graph
from ai_research_agent.graph.advanced import (
    MAX_RESEARCH_LOOPS,
    route_after_analyze,
    web_research_node,
)
from ai_research_agent.graph.state import ResearchState


def test_router_loops_when_evidence_thin() -> None:
    state: ResearchState = {"question": "q", "evidence": ["a"], "iterations": 0}
    assert route_after_analyze(state) == "more"


def test_router_stops_at_iteration_cap() -> None:
    state: ResearchState = {"question": "q", "evidence": ["a"], "iterations": MAX_RESEARCH_LOOPS}
    assert route_after_analyze(state) == "done"


def test_router_stops_when_evidence_sufficient() -> None:
    state: ResearchState = {"question": "q", "evidence": ["x"] * 50, "iterations": 0}
    assert route_after_analyze(state) == "done"


def test_web_research_node_uses_offline_fixture() -> None:
    from ai_research_agent.schemas.research import ResearchPlan, SubQuestion

    plan = ResearchPlan(
        objective="o",
        sub_questions=[SubQuestion(question="AI coding agents", rationale="r", needs_web=True)],
    )
    update = web_research_node({"question": "q", "plan": plan})
    assert update["evidence"]
    assert all(e.startswith("[web:") for e in update["evidence"])


def test_branching_graph_runs_and_terminates() -> None:
    state = build_branching_graph().invoke({"question": "Enter the AI coding agent market?"})
    assert state["answer"]
    assert state.get("iterations", 0) <= MAX_RESEARCH_LOOPS  # cycle terminated
    # Evidence came from both web (web:) and internal (#) sources.
    assert any(e.startswith("[web:") for e in state["evidence"])
    assert any("#" in e for e in state["evidence"])


def test_branching_graph_has_cycle_edge() -> None:
    graph = build_branching_graph().get_graph()
    assert {"web_research", "internal_research", "analyze", "more_research"} <= set(graph.nodes)
