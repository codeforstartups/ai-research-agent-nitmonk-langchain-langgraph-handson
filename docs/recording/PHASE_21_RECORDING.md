# Phase 21 — MCP + LangGraph

> **Git tag:** `phase-21`
> **Checkout:** `git checkout phase-21`
> **One-liner:** graph nodes gather evidence by calling MCP servers.

## Where we are

The graph and MCP were separate. Phase 21 joins them: the research node calls
the **MCP client**, so the graph's capabilities are whatever servers are
connected — with no idea how any of them work. That's the modularity payoff.

## Concepts explained

- **MCP inside a node**: `await client.call_tool("knowledge__search_knowledge", …)`
  and `web_research__search_web` from within a graph node.
- **Async graphs**: MCP is async, so the node is `async def` and we run
  `await graph.ainvoke(...)` inside the open `MultiMCPClient` context.
- **Graceful capability use**: the node checks which tools exist and uses only
  those — register a web server and web evidence appears; omit it and the graph
  still runs on knowledge alone.
- **Separation of concerns**: the graph orchestrates; MCP provides capability.
  Add a server → new capability, zero graph changes.

## Files created / modified

```text
src/ai_research_agent/graph/mcp_graph.py    # build_mcp_graph (async node)
examples/09_langgraph/mcp_graph.py
tests/agents/test_mcp_graph.py
```

## Commands to run (on camera)

```bash
uv run pytest tests/agents/test_mcp_graph.py -q
uv run python examples/09_langgraph/mcp_graph.py
```

## Expected output

The graph gathers evidence blocks tagged `[knowledge]` and `[web]` (both via
MCP), lists sources `['knowledge-mcp', 'web-mcp', …]`, and synthesizes an answer.

## Demo moment

Run once with all three servers, then rerun connecting *only* the knowledge
server — the graph still works, just without web evidence. "Capabilities are
plug-in. The graph never changed."

## Next phase

`phase-22` — a supervisor with specialist subagents.
