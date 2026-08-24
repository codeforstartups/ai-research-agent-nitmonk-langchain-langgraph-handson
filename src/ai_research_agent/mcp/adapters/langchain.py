"""Adapt MCP tools into LangChain tools.

Why not ``langchain-mcp-adapters``? That package currently pins ``mcp<2`` (the
old FastMCP API), whereas this course targets the **official MCP Python SDK v2**
(``MCPServer``). Installing it would downgrade the SDK and break every server we
built. So we write the adapter ourselves — which is also the clearest way to
teach what "adapting a tool interface" actually means:

    discovered MCP tool (name + description + JSON schema)
        → a LangChain ``StructuredTool`` whose coroutine calls the MCP client.

The adapted tools are async and remain valid only while the client's sessions
are open (use them inside ``async with MultiMCPClient(...) as client:``).
"""

from __future__ import annotations

from typing import Any

from langchain_core.tools import StructuredTool
from pydantic import Field, create_model

from ai_research_agent.mcp.client.multi_client import DiscoveredTool, MultiMCPClient

_JSON_TO_PY: dict[str, type] = {
    "string": str,
    "integer": int,
    "number": float,
    "boolean": bool,
    "array": list,
    "object": dict,
}


def _args_model(tool: DiscoveredTool):
    """Build a Pydantic args model from a tool's JSON input schema."""
    schema = tool.input_schema or {}
    properties: dict[str, Any] = schema.get("properties", {})
    required = set(schema.get("required", []))

    fields: dict[str, tuple] = {}
    for name, spec in properties.items():
        py_type = _JSON_TO_PY.get(spec.get("type", "string"), str)
        description = spec.get("description", "")
        if name in required:
            fields[name] = (py_type, Field(description=description))
        else:
            default = spec.get("default", None)
            fields[name] = (py_type, Field(default=default, description=description))

    model_name = f"{tool.server}_{tool.name}_Args"
    return create_model(model_name, **fields) if fields else create_model(model_name)


def mcp_tool_to_langchain(client: MultiMCPClient, tool: DiscoveredTool) -> StructuredTool:
    """Wrap a single discovered MCP tool as a LangChain StructuredTool."""

    async def _call(**kwargs: Any) -> str:
        return await client.call_tool(tool.qualified_name, kwargs)

    return StructuredTool(
        name=tool.qualified_name,
        description=tool.description,
        args_schema=_args_model(tool),
        coroutine=_call,
    )


def to_langchain_tools(client: MultiMCPClient) -> list[StructuredTool]:
    """Adapt every tool discovered by ``client`` into LangChain tools."""
    return [mcp_tool_to_langchain(client, t) for t in client.list_tools()]
