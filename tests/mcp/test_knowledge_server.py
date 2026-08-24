"""Phase 14 — Knowledge MCP server (in-memory)."""

from __future__ import annotations

import pytest
from mcp import Client

from ai_research_agent.mcp.servers import build_knowledge_server


@pytest.fixture
def server():
    return build_knowledge_server()


async def test_search_knowledge_returns_citations(server) -> None:
    async with Client(server) as client:
        result = await client.call_tool(
            "search_knowledge", {"query": "enterprise requirements SOC2"}
        )
    assert "#" in result.content[0].text  # [source#chunk] citation


async def test_list_sources_includes_sample_docs(server) -> None:
    async with Client(server) as client:
        result = await client.call_tool("list_sources", {})
    assert "enterprise_requirements.md" in result.content[0].text


async def test_get_document_returns_full_text(server) -> None:
    async with Client(server) as client:
        result = await client.call_tool("get_document", {"source": "market_overview.md"})
    assert "segmentation" in result.content[0].text.lower()


async def test_document_resource(server) -> None:
    async with Client(server) as client:
        result = await client.read_resource("knowledge://document/competitors.md")
    assert "moat" in result.contents[0].text.lower()


async def test_missing_document_is_handled(server) -> None:
    async with Client(server) as client:
        result = await client.call_tool("get_document", {"source": "nope.md"})
    assert "No document" in result.content[0].text
