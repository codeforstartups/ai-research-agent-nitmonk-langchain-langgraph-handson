"""Phase 23 — fact-checker + critic verification cycle."""

from __future__ import annotations

from ai_research_agent.graph.verification import (
    MAX_VERIFY_LOOPS,
    build_verified_graph,
    fact_check_node,
    route_fact_check,
)
from ai_research_agent.schemas.agents import Finding, Verdict


def test_fact_check_passes_with_grounded_answer() -> None:
    state = {
        "question": "q",
        "answer": "Enterprises require SOC2 and audit logs [enterprise.md#0].",
        "findings": [Finding(facet="enterprise", summary="...", sources=["enterprise.md#0"])],
    }
    verdict = fact_check_node(state)["verdict"]
    assert verdict.passed is True
    assert verdict.issues == []


def test_fact_check_fails_without_sources() -> None:
    verdict = fact_check_node({"question": "q", "answer": "A long enough answer text"})["verdict"]
    assert verdict.passed is False
    assert any("No sources" in i for i in verdict.issues)


def test_router_fails_then_passes_at_cap() -> None:
    failing = {"verdict": Verdict(passed=False, issues=["x"]), "iterations": 0}
    assert route_fact_check(failing) == "fail"
    capped = {"verdict": Verdict(passed=False, issues=["x"]), "iterations": MAX_VERIFY_LOOPS}
    assert route_fact_check(capped) == "pass"


def test_verified_graph_end_to_end() -> None:
    state = build_verified_graph().invoke({"question": "Enter the AI coding-agent market?"})
    assert state["verdict"].passed is True
    assert state["answer"]
    assert state.get("iterations", 0) <= MAX_VERIFY_LOOPS


def test_verified_graph_has_check_and_critic_nodes() -> None:
    nodes = set(build_verified_graph().get_graph().nodes)
    assert {"fact_check", "revise", "critic"} <= nodes
