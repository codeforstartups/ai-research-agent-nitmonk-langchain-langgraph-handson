"""Phase 18 — the LangGraph research graph and its nodes."""

from __future__ import annotations

from ai_research_agent.graph import build_research_graph, run_research
from ai_research_agent.graph.nodes import research_node, synthesize_node
from ai_research_agent.graph.state import ResearchState
from ai_research_agent.schemas.research import ResearchPlan, SubQuestion


def test_graph_runs_end_to_end() -> None:
    state = run_research("What do enterprises require from AI coding agents?")
    assert state["plan"].sub_questions  # planned
    assert state["evidence"]  # retrieved
    assert state["answer"]  # synthesized
    assert state["sources"]  # cited


def test_research_node_gathers_evidence_from_plan() -> None:
    plan = ResearchPlan(
        objective="x",
        sub_questions=[SubQuestion(question="SOC2 enterprise requirements", rationale="r")],
    )
    state: ResearchState = {"question": "q", "plan": plan}
    update = research_node(state)
    assert update["evidence"]
    assert all("#" in e for e in update["evidence"])  # cited


def test_synthesize_node_uses_evidence() -> None:
    state: ResearchState = {
        "question": "q",
        "evidence": ["[a.md#0] enterprises require SOC2 and audit logs"],
    }
    update = synthesize_node(state)
    assert isinstance(update["answer"], str) and update["answer"]


def test_graph_compiles_with_expected_nodes() -> None:
    graph = build_research_graph()
    node_names = set(graph.get_graph().nodes)
    assert {"plan", "research", "synthesize"} <= node_names
