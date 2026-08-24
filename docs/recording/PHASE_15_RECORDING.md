# Phase 15 — Multi-MCP client

> **Git tag:** `phase-15`
> **Checkout:** `git checkout phase-15`
> **One-liner:** connect to many MCP servers at once, discover their tools, and
> route calls without name collisions.

## Where we are

We have three servers. Phase 15 builds the **client** that binds them into one
capability surface — the layer the agent will actually call.

```text
                    AGENT
                      │
                 MultiMCPClient
        ┌─────────────┼──────────────┐
        ▼             ▼              ▼
   analysis     web_research     knowledge
```

## Concepts explained

- **Dynamic discovery**: the client asks each server `list_tools()` at runtime —
  it never assumes servers share a capability set (a server may have tools but
  no prompts).
- **Namespacing**: tools are exposed as `server__tool` so `analysis__calculate`
  and a hypothetical `other__calculate` can coexist.
- **Routing**: `call_tool("web_research__search_web", …)` dispatches to the
  right session.
- **In-memory now, transports later**: the SDK's `Client(server_object)` gives
  hermetic in-process sessions; the same client works over stdio/HTTP by passing
  a transport instead.
- **Session lifecycle**: `AsyncExitStack` opens/closes every server session
  cleanly.

## Files created / modified

```text
src/ai_research_agent/mcp/client/multi_client.py   # MultiMCPClient, DiscoveredTool
src/ai_research_agent/mcp/client/__init__.py
examples/07_mcp/multi_client.py
tests/mcp/test_mcp_client.py
```

## Commands to run (on camera)

```bash
uv run pytest tests/mcp -q
uv run python examples/07_mcp/multi_client.py
```

## Expected output

A list of all discovered tools with `server__tool` names, then one call routed
to each server: `analysis__calculate` → `1800.0`, `web_research__search_web` →
untrusted-wrapped results, `knowledge__list_sources` → the sample docs.

## Demo moment

Show the aggregated tool list — capabilities from three independent servers,
uniformly named — and make the point: adding a new capability to the agent is
now "stand up a server and register it", not "edit the agent".

## Next phase

`phase-16` — adapt MCP tools into LangChain tools (langchain-mcp-adapters).
