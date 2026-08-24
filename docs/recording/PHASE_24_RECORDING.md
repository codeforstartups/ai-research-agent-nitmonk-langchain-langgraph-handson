# Phase 24 — Deep Agents

> **Git tag:** `phase-24`
> **Checkout:** `git checkout phase-24`
> **One-liner:** rebuild the agent on the Deep Agents harness and compare it to
> the hand-built graph.

## Where we are

Phases 18-23 assembled the agent from LangGraph primitives by hand. Phase 24
uses **Deep Agents** — a harness that ships planning, a filesystem, subagents,
memory, HITL and streaming out of the box — and gives it the *same*
capabilities so we can compare approaches.

```text
LangChain  → building blocks
LangGraph  → stateful runtime you assemble yourself
Deep Agents → a pre-assembled harness on top of both
```

> Deep Agents requires Python ≥3.11 — which is why the repo moved to 3.11 in the
> commit before this phase.

## Concepts explained

- **`create_deep_agent(model, tools, system_prompt, subagents=...)`** returns a
  compiled LangGraph with the harness middleware baked in.
- **Planning**: a built-in todo tool for multi-step work.
- **Filesystem / context management**: scratch files so long runs don't overflow
  the context window.
- **Subagents**: `SubAgent{name, description, system_prompt, tools}` — the lead
  delegates facet research (market / competitor / enterprise), mirroring phase 22
  but without wiring the graph ourselves.
- **When to reach for the harness vs the hand-built graph**: control &
  transparency (graph) vs speed & batteries-included (harness).

## Files created / modified

```text
src/ai_research_agent/deep_agent/agent.py    # build_research_deep_agent
src/ai_research_agent/deep_agent/__init__.py
examples/10_deep_agents/deep_agent.py
tests/agents/test_deep_agent.py
pyproject.toml                                # + deepagents (needs py3.11)
```

## Commands to run (on camera)

```bash
uv run pytest tests/agents/test_deep_agent.py -q
uv run python examples/10_deep_agents/deep_agent.py     # needs a key to plan/delegate
```

## Expected output

The agent's node list (model / tools / middleware) and its three subagents.
Offline it returns the stub; with a key it plans with todos, delegates to
specialists, calls tools, and writes a cited report.

## Demo moment

Put phase 23's hand-built `verified_graph` next to this harness agent: "same
capabilities — planning, subagents, delegation — one we built explicitly, one we
got from a harness. Now you can choose per project." Then run with a key and
watch it write and check off todos.

## Next phase

`phase-25` — Langfuse observability across the whole run.
