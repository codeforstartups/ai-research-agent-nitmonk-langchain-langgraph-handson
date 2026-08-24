# Phase 14 — Knowledge MCP

> **Git tag:** `phase-14`
> **Checkout:** `git checkout phase-14`
> **One-liner:** expose the internal RAG stack as its own MCP server.

## Where we are

We now have two capability sources — live web (phase 13) and internal knowledge.
Phase 14 wraps our RAG retriever as a **Knowledge MCP server**, so the agent
consults internal docs through the *same* discover-and-call mechanism it uses
for the web. Two very different context sources, one uniform interface.

```text
Agent → MCP Client → ┬ Web Research MCP → Thordata → Internet
                     └ Knowledge MCP    → Hybrid RAG → internal docs
```

## Concepts explained

- **RAG as a server, not a hard-wired stage**: the model *chooses* to search
  internal knowledge, just like it chooses to search the web.
- **Tools + resource + prompt over the corpus**: `search_knowledge`,
  `list_sources`, `get_document`; `knowledge://document/{source}`;
  `summarize_source`.
- **Two knowledge paths, one protocol** — the "external live vs internal
  persistent" distinction from the architecture, made concrete.

## Files created / modified

```text
src/ai_research_agent/mcp/servers/knowledge.py     # build_knowledge_server
src/ai_research_agent/mcp/servers/__init__.py      # export it
examples/07_mcp/knowledge_server.py
tests/mcp/test_knowledge_server.py
pyproject.toml                                      # ruff: mcp = third-party
```

## Commands to run (on camera)

```bash
uv run pytest tests/mcp -q
uv run python examples/07_mcp/knowledge_server.py
npx @modelcontextprotocol/inspector uv run python examples/07_mcp/knowledge_server.py
```

## Expected output

In the Inspector, `list_sources` returns the sample docs; `search_knowledge("SOC2
enterprise requirements")` returns cited passages; the
`knowledge://document/competitors.md` resource returns the full note.

## Demo moment

Put the Web Research Inspector and the Knowledge Inspector side by side: same
protocol, same call shape, radically different backends (Thordata vs a vector
index). That symmetry is why the agent can treat them uniformly — which we wire
up with a multi-server client next.

## Next phase

`phase-15` — the multi-MCP client (connect many servers, dynamic discovery).
