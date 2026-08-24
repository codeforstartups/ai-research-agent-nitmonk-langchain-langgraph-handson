"""Phase 21 — a graph whose nodes call MCP capabilities."""

from __future__ import annotations

from ai_research_agent.graph.mcp_graph import build_mcp_graph
from ai_research_agent.mcp.client import MultiMCPClient, default_servers


async def test_graph_gathers_evidence_via_mcp() -> None:
    async with MultiMCPClient(default_servers()) as client:
        graph = build_mcp_graph(client)
        state = await graph.ainvoke({"question": "Enter the AI coding-agent market?"})

    assert state["answer"]
    # Evidence came through MCP: both the knowledge and web servers contributed.
    assert any(e.startswith("[knowledge]") for e in state["evidence"])
    assert any(e.startswith("[web]") for e in state["evidence"])
    assert "knowledge-mcp" in state["sources"]
    assert "web-mcp" in state["sources"]


async def test_graph_runs_with_only_knowledge_server() -> None:
    from ai_research_agent.mcp.servers import build_knowledge_server

    async with MultiMCPClient({"knowledge": build_knowledge_server()}) as client:
        graph = build_mcp_graph(client)
        state = await graph.ainvoke({"question": "enterprise requirements"})
    # Gracefully uses only what's available (no web server registered).
    assert any(e.startswith("[knowledge]") for e in state["evidence"])
    assert all(not e.startswith("[web]") for e in state["evidence"])
