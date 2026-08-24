"""Phase 24 — the Deep Agent harness."""

from __future__ import annotations

from langgraph.graph.state import CompiledStateGraph

from ai_research_agent.deep_agent import build_research_deep_agent


def test_deep_agent_builds() -> None:
    agent = build_research_deep_agent()
    assert isinstance(agent, CompiledStateGraph)


def test_deep_agent_has_model_and_tools_nodes() -> None:
    nodes = set(build_research_deep_agent().get_graph().nodes)
    assert "model" in nodes
    assert "tools" in nodes  # the built-in tool loop


def test_deep_agent_runs_offline_without_crashing() -> None:
    agent = build_research_deep_agent()
    out = agent.invoke(
        {"messages": [{"role": "user", "content": "What do enterprises require?"}]}
    )
    assert out["messages"]  # produced at least one response message
