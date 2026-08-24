"""Phase 22 — supervisor + specialist subagents."""

from __future__ import annotations

from ai_research_agent.graph.supervisor import build_supervisor_graph
from ai_research_agent.schemas.agents import Finding
from ai_research_agent.subagents import market_specialist


def test_specialist_returns_single_typed_finding() -> None:
    update = market_specialist({"question": "market segmentation for AI coding agents"})
    findings = update["findings"]
    assert len(findings) == 1
    assert isinstance(findings[0], Finding)
    assert findings[0].facet == "market"
    assert findings[0].sources


def test_supervisor_runs_all_specialists_in_parallel() -> None:
    state = build_supervisor_graph().invoke({"question": "Enter the AI coding-agent market?"})
    facets = {f.facet for f in state["findings"]}
    assert facets == {"market", "competitor", "enterprise"}
    assert state["answer"]
    assert state["sources"]


def test_supervisor_graph_has_specialist_nodes() -> None:
    nodes = set(build_supervisor_graph().get_graph().nodes)
    assert {"market_agent", "competitor_agent", "enterprise_agent", "aggregate"} <= nodes
