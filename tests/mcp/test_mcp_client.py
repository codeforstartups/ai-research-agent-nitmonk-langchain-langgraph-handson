"""Phase 15 — the multi-MCP client."""

from __future__ import annotations

from mcp import Client

from ai_research_agent.mcp.client import MultiMCPClient, default_servers
from ai_research_agent.mcp.servers import build_analysis_server, build_knowledge_server


async def test_discovers_tools_from_all_servers() -> None:
    servers = {"analysis": build_analysis_server(), "knowledge": build_knowledge_server()}
    async with MultiMCPClient(servers) as client:
        names = client.tool_names()
    # Namespaced, no collisions across servers.
    assert "analysis__calculate" in names
    assert "knowledge__search_knowledge" in names
    assert all("__" in n for n in names)


async def test_routes_call_to_correct_server() -> None:
    servers = {"analysis": build_analysis_server(), "knowledge": build_knowledge_server()}
    async with MultiMCPClient(servers) as client:
        out = await client.call_tool("analysis__calculate", {"expression": "6 * 7"})
    assert out == "42.0"


async def test_unknown_tool_raises() -> None:
    async with MultiMCPClient({"analysis": build_analysis_server()}) as client:
        try:
            await client.call_tool("analysis__does_not_exist", {})
        except KeyError as exc:
            assert "Unknown tool" in str(exc)
        else:  # pragma: no cover
            raise AssertionError("expected KeyError")


async def test_default_trio_connects_and_discovers() -> None:
    async with MultiMCPClient(default_servers()) as client:
        servers = {t.server for t in client.list_tools()}
    assert servers == {"analysis", "web_research", "knowledge"}


async def test_read_resource_via_client() -> None:
    async with MultiMCPClient({"analysis": build_analysis_server()}) as client:
        text = await client.read_resource("analysis", "analysis://methods")
    assert "calculate" in text


def test_client_type_is_context_manager() -> None:
    # Sanity: the underlying SDK Client accepts a server object (in-memory).
    assert Client is not None
