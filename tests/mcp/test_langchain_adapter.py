"""Phase 16 — adapting MCP tools into LangChain tools."""

from __future__ import annotations

from langchain_core.tools import StructuredTool

from ai_research_agent.mcp.adapters import to_langchain_tools
from ai_research_agent.mcp.client import MultiMCPClient
from ai_research_agent.mcp.servers import build_analysis_server


async def test_adapter_produces_langchain_tools() -> None:
    async with MultiMCPClient({"analysis": build_analysis_server()}) as client:
        tools = to_langchain_tools(client)
        names = {t.name for t in tools}
        assert all(isinstance(t, StructuredTool) for t in tools)
        assert "analysis__calculate" in names


async def test_adapted_tool_args_schema_reflects_inputs() -> None:
    async with MultiMCPClient({"analysis": build_analysis_server()}) as client:
        tools = {t.name: t for t in to_langchain_tools(client)}
        schema = tools["analysis__calculate"].args_schema.model_json_schema()
        assert "expression" in schema["properties"]


async def test_adapted_tool_ainvoke_routes_to_mcp() -> None:
    async with MultiMCPClient({"analysis": build_analysis_server()}) as client:
        tools = {t.name: t for t in to_langchain_tools(client)}
        result = await tools["analysis__calculate"].ainvoke({"expression": "9 * 9"})
    assert result == "81.0"
