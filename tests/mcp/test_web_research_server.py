"""Phase 13 — Web Research MCP server (offline fixture path)."""

from __future__ import annotations

import pytest

from ai_research_agent.mcp.servers.web_research import (
    ThordataClient,
    build_web_research_server,
)
from ai_research_agent.mcp.servers.web_research.server import UNTRUSTED_BANNER
from mcp import Client


@pytest.fixture
def server():
    # No API key -> deterministic offline fixture.
    return build_web_research_server(ThordataClient(api_key=None))


async def test_search_web_returns_wrapped_results(server) -> None:
    async with Client(server) as client:
        result = await client.call_tool("search_web", {"query": "AI coding agents", "count": 3})
    text = result.content[0].text
    assert UNTRUSTED_BANNER in text  # untrusted-content boundary is applied
    assert "https://" in text  # URLs preserved


async def test_scrape_url_is_wrapped(server) -> None:
    async with Client(server) as client:
        result = await client.call_tool("scrape_url", {"url": "https://example.com"})
    assert UNTRUSTED_BANNER in result.content[0].text


async def test_sources_resource_reflects_searches(server) -> None:
    async with Client(server) as client:
        await client.call_tool("search_web", {"query": "AI coding agents", "count": 3})
        listing = await client.read_resource("research://sources")
        one = await client.read_resource("research://source/1")
    assert "http" in listing.contents[0].text
    assert "url:" in one.contents[0].text


async def test_prompts_available(server) -> None:
    async with Client(server) as client:
        names = {p.name for p in (await client.list_prompts()).prompts}
        rendered = await client.get_prompt("research_company", {"company": "Cursor"})
    assert names == {"research_company", "analyze_market"}
    assert "Cursor" in rendered.messages[0].content.text


def test_thordata_client_offline_fixture() -> None:
    results = ThordataClient(api_key=None).search("anything", count=2)
    assert len(results) == 2
    assert all(r.url.startswith("http") for r in results)
    assert all(r.source == "offline-fixture" for r in results)
