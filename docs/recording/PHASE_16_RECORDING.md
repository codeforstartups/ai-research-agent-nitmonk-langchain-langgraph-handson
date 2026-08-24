# Phase 16 — MCP + LangChain

> **Git tag:** `phase-16`
> **Checkout:** `git checkout phase-16`
> **One-liner:** turn discovered MCP tools into LangChain tools the model can
> call.

## Where we are

The multi-client discovers MCP tools; LangChain models call `StructuredTool`s.
Phase 16 bridges them: every MCP tool becomes a LangChain tool, so MCP
capabilities drop straight into the tool-calling loop (and, soon, the agent).

## Concepts explained

- **Adapting an interface**: a discovered MCP tool (name + description + JSON
  schema) → a `StructuredTool` whose coroutine calls the MCP client.
- **JSON schema → Pydantic args**: we build the tool's `args_schema` from the
  server's advertised input schema, so validation and tool-calling "just work".
- **Why hand-rolled, not `langchain-mcp-adapters`**: that package pins `mcp<2`
  (old FastMCP); installing it would downgrade the SDK v2 we teach and break our
  servers. A real, honest integration constraint — shown, not hidden.
- **Lifecycle**: adapted tools are async and valid only while the client's
  sessions are open.

## Files created / modified

```text
src/ai_research_agent/mcp/adapters/langchain.py    # to_langchain_tools
src/ai_research_agent/mcp/adapters/__init__.py
src/ai_research_agent/mcp/client/multi_client.py    # fix: input_schema (v2)
examples/07_mcp/langchain_adapter.py
tests/mcp/test_langchain_adapter.py
```

## Commands to run (on camera)

```bash
uv run pytest tests/mcp -q
uv run python examples/07_mcp/langchain_adapter.py
```

## Expected output

All MCP tools listed as LangChain tools (`analysis__calculate`,
`web_research__search_web`, `knowledge__search_knowledge`, …). Invoking
`analysis__calculate` directly returns `1800.0`. With a key, the model *chooses*
an MCP tool for a knowledge query.

## Demo moment

Open `adapters/langchain.py` and show the ~15 lines that convert a JSON schema
into a Pydantic args model. "This is all an 'MCP adapter' is — and now every
server we add is instantly available to any LangChain model or graph."

## Next phase

`phase-17` — a LangChain tool-calling agent over these tools.
