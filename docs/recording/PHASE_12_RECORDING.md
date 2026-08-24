# Phase 12 — MCP foundation

> **Git tag:** `phase-12`
> **Checkout:** `git checkout phase-12`
> **One-liner:** stand up a first MCP server exposing all three primitives, and
> talk to it.

## Where we are

Phase 11 gave the model local `@tool` functions. MCP generalises that: instead
of importing tools, the agent connects to **servers** over a protocol and
**discovers** their capabilities. This is the external capability boundary — and
the reason the final agent can mix web, knowledge, and analysis servers freely.

```text
Application → MCP Client → MCP Protocol → MCP Server → Tool
```

## Concepts explained

- **The three primitives, by who controls them**:
  - **tools** — model-controlled (`calculate`, `percentage_change`),
  - **resources** — application-controlled context (`analysis://methods`),
  - **prompts** — user-controlled templates (`compare_competitors`).
- **Official MCP Python SDK v2** — `MCPServer` with `@server.tool/resource/prompt`.
- **Transport (stdio)** — the server reads/writes JSON-RPC over stdin/stdout;
  the host launches it as a subprocess.
- **In-memory testing** — `Client(server)` connects directly to a server object,
  no subprocess or ports. Fast, hermetic tests.
- **MCP Inspector** — the interactive dev tool for browsing/calling primitives.

## Files created / modified

```text
src/ai_research_agent/mcp/__init__.py
src/ai_research_agent/mcp/servers/__init__.py
src/ai_research_agent/mcp/servers/analysis.py    # tool + resource + prompt
examples/07_mcp/basic_server.py                  # stdio entry point
tests/mcp/test_analysis_server.py                # in-memory client tests
pyproject.toml                                    # + mcp (SDK v2)
```

## Commands to run (on camera)

```bash
uv run pytest tests/mcp -q

# Run the server (it waits for a host on stdio):
uv run python examples/07_mcp/basic_server.py     # Ctrl-C to stop

# Explore it with the Inspector (Node required):
npx @modelcontextprotocol/inspector uv run python examples/07_mcp/basic_server.py
```

## Expected output

The tests connect a client to the server in-memory and verify it lists two
tools, one resource, and one prompt, and that `calculate` returns `1800.0`. In
the Inspector you can click `calculate`, pass `12000 * 0.15`, and see `1800.0`.

## Demo moment

Open the Inspector, browse the **Tools / Resources / Prompts** tabs, call
`calculate`, then read the `analysis://methods` resource. Narrate the control
distinction: the model calls tools, the app reads resources, the user picks
prompts.

## Next phase

`phase-13` — the Web Research MCP server, powered by Thordata.
