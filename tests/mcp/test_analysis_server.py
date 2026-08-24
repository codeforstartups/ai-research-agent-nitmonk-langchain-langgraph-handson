"""Phase 12 — the Analysis MCP server, tested in-memory.

The MCP SDK lets a Client connect directly to a server object — no subprocess,
no ports — so server tests are fast and hermetic.
"""

from __future__ import annotations

import pytest
from mcp import Client

from ai_research_agent.mcp.servers import build_analysis_server


@pytest.fixture
def server():
    return build_analysis_server()


async def test_lists_all_three_primitives(server) -> None:
    async with Client(server) as client:
        tools = {t.name for t in (await client.list_tools()).tools}
        resources = {str(r.uri) for r in (await client.list_resources()).resources}
        prompts = {p.name for p in (await client.list_prompts()).prompts}

    assert tools == {"calculate", "percentage_change"}
    assert "analysis://methods" in resources
    assert "compare_competitors" in prompts


async def test_call_tool_calculate(server) -> None:
    async with Client(server) as client:
        result = await client.call_tool("calculate", {"expression": "12000 * 0.15"})
    assert result.content[0].text == "1800.0"


async def test_call_tool_percentage_change(server) -> None:
    async with Client(server) as client:
        result = await client.call_tool(
            "percentage_change", {"old_value": 100, "new_value": 150}
        )
    assert result.content[0].text == "+50.00%"


async def test_read_resource(server) -> None:
    async with Client(server) as client:
        result = await client.read_resource("analysis://methods")
    assert "calculate" in result.contents[0].text


async def test_get_prompt(server) -> None:
    async with Client(server) as client:
        result = await client.get_prompt(
            "compare_competitors",
            {"company_a": "Cursor", "company_b": "Copilot", "dimension": "reliability"},
        )
    rendered = result.messages[0].content.text
    assert "Cursor" in rendered and "Copilot" in rendered
