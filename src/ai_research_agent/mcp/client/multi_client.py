"""A client that connects to several MCP servers at once.

A real agent draws capabilities from multiple servers (analysis + web + internal
knowledge). This client:

* opens a session to each server,
* **discovers** each server's capabilities at runtime (never assuming they're
  the same — a server may expose tools but no prompts),
* aggregates tools under **namespaced** names (``server__tool``) so names from
  different servers can never collide,
* routes ``call_tool`` / ``read_resource`` / ``get_prompt`` to the right server.

Servers are connected **in-memory** here (``Client(server_object)``), which is
also how tests run. The identical client works against stdio/HTTP servers by
passing transports instead of server objects.
"""

from __future__ import annotations

from contextlib import AsyncExitStack
from dataclasses import dataclass

from mcp import Client
from mcp.server import MCPServer

NAMESPACE_SEP = "__"


@dataclass
class DiscoveredTool:
    """A tool discovered on some server, with its namespaced name."""

    server: str
    name: str  # bare tool name on the server
    description: str
    input_schema: dict

    @property
    def qualified_name(self) -> str:
        return f"{self.server}{NAMESPACE_SEP}{self.name}"


def _result_text(result) -> str:
    """Flatten an MCP tool/resource result into text."""
    parts = []
    for block in getattr(result, "content", []) or []:
        text = getattr(block, "text", None)
        if text is not None:
            parts.append(text)
    return "\n".join(parts)


class MultiMCPClient:
    """Manage sessions to multiple MCP servers behind one interface."""

    def __init__(self, servers: dict[str, MCPServer]) -> None:
        self._servers = servers
        self._clients: dict[str, Client] = {}
        self._tool_index: dict[str, DiscoveredTool] = {}
        self._stack: AsyncExitStack | None = None

    async def __aenter__(self) -> MultiMCPClient:
        self._stack = AsyncExitStack()
        for name, server in self._servers.items():
            client = await self._stack.enter_async_context(Client(server))
            self._clients[name] = client
        await self._discover()
        return self

    async def __aexit__(self, *exc) -> None:
        if self._stack is not None:
            await self._stack.aclose()
        self._clients.clear()
        self._tool_index.clear()

    async def _discover(self) -> None:
        for server_name, client in self._clients.items():
            listing = await client.list_tools()
            for tool in listing.tools:
                # MCP SDK v2 exposes the JSON schema as `input_schema` (snake).
                schema = getattr(tool, "input_schema", None) or getattr(tool, "inputSchema", {})
                discovered = DiscoveredTool(
                    server=server_name,
                    name=tool.name,
                    description=tool.description or "",
                    input_schema=schema or {},
                )
                self._tool_index[discovered.qualified_name] = discovered

    # -- discovery ----------------------------------------------------------
    def list_tools(self) -> list[DiscoveredTool]:
        return list(self._tool_index.values())

    def tool_names(self) -> list[str]:
        return sorted(self._tool_index)

    # -- invocation ---------------------------------------------------------
    async def call_tool(self, qualified_name: str, arguments: dict | None = None) -> str:
        tool = self._tool_index.get(qualified_name)
        if tool is None:
            raise KeyError(f"Unknown tool {qualified_name!r}. Known: {self.tool_names()}")
        client = self._clients[tool.server]
        result = await client.call_tool(tool.name, arguments or {})
        return _result_text(result)

    async def read_resource(self, server: str, uri: str) -> str:
        result = await self._clients[server].read_resource(uri)
        return "\n".join(getattr(c, "text", "") for c in result.contents)

    async def get_prompt(self, server: str, name: str, arguments: dict | None = None) -> str:
        result = await self._clients[server].get_prompt(name, arguments or {})
        return "\n".join(
            getattr(m.content, "text", "") for m in result.messages
        )


def default_servers() -> dict[str, MCPServer]:
    """The standard trio of servers the agent connects to."""
    from ai_research_agent.mcp.servers import (
        build_analysis_server,
        build_knowledge_server,
        build_web_research_server,
    )

    return {
        "analysis": build_analysis_server(),
        "web_research": build_web_research_server(),
        "knowledge": build_knowledge_server(),
    }
