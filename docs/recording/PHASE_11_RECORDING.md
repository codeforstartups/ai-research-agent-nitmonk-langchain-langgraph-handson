# Phase 11 — LangChain tools

> **Git tag:** `phase-11`
> **Checkout:** `git checkout phase-11`
> **One-liner:** give the model callable tools with typed schemas, and run the
> tool-calling loop.

## Where we are

So far the model only produced text. Phase 11 gives it **actions**: typed tools
it can choose to call. This is the pivot from "a RAG chain we drive" to "an agent
that decides".

## Concepts explained

- **`@tool`**: a function's signature + docstring become the schema the model
  reads to decide *when* and *how* to call it.
- **Tools are authority**: `calculate` parses a restricted arithmetic AST — no
  `eval()`. Granting a tool grants power; validate its inputs. (Same discipline
  as MCP + untrusted content later.)
- **`bind_tools` + the loop**: model requests calls → we execute → feed
  `ToolMessage` results back → model finalises.
- **Retrieval as a tool**: `search_knowledge` makes RAG *model-controlled*.

## Files created / modified

```text
src/ai_research_agent/tools/analysis.py     # calculate (safe), percentage_change
src/ai_research_agent/tools/knowledge.py    # search_knowledge (RAG as a tool)
src/ai_research_agent/tools/__init__.py     # ALL_TOOLS
examples/06_tools/tool_calling.py
tests/unit/test_tools.py
```

## Commands to run (on camera)

```bash
uv run pytest -q
uv run python examples/06_tools/tool_calling.py
```

## Expected output

Tools invoked directly (calculate → `1800.0`, search_knowledge → cited passage).
With a key, the tool-calling loop shows the model *choosing* `calculate`, us
executing it, and the model using the result in its final answer.

## Demo moment

Run `test_calculate_rejects_unsafe_input` and show `__import__('os').system(...)`
returning `Error`, not executing. Say it plainly: "a tool is authority; we
validate every input — this mindset is exactly what MCP formalises next."

## Next phase

`phase-12` — MCP foundation: server, tool, resource, prompt, Inspector.
