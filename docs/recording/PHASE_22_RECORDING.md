# Phase 22 — Subagents (supervisor + specialists)

> **Git tag:** `phase-22`
> **Checkout:** `git checkout phase-22`
> **One-liner:** split research across focused specialist subagents coordinated
> by a supervisor.

## Where we are

One generalist doing market + tech + competition at once produces mushy
evidence. Phase 22 introduces **specialists** — each with a narrow remit and a
typed `Finding` — dispatched in parallel by a **supervisor** and merged by an
aggregator.

```text
supervisor → [market_agent · competitor_agent · enterprise_agent] → aggregate → END
```

## Concepts explained

- **Specialist subagent**: defined responsibility, focused retrieval, typed
  output (`Finding{facet, summary, sources}`), evaluable on its own facet.
- **Supervisor**: coordinates — here it plans and fans out to all specialists.
- **Parallel + reducer**: specialists run concurrently and each appends its
  `Finding`; the aggregator reads them all.
- **Factory pattern**: `make_specialist(facet, focus, system)` — new specialists
  are one line.

## Files created / modified

```text
src/ai_research_agent/schemas/agents.py          # Finding, Verdict
src/ai_research_agent/subagents/specialists.py    # make_specialist + the three
src/ai_research_agent/subagents/__init__.py
src/ai_research_agent/graph/supervisor.py         # build_supervisor_graph
src/ai_research_agent/graph/state.py               # + findings, verdict
examples/09_langgraph/supervisor.py
tests/agents/test_supervisor.py
```

## Commands to run (on camera)

```bash
uv run pytest tests/agents/test_supervisor.py -q
uv run python examples/09_langgraph/supervisor.py
```

## Expected output

The graph diagram, three findings (`market`, `competitor`, `enterprise`) each
with its own sources, and an aggregated answer citing the union of sources.

## Demo moment

Show the three specialists producing separate, focused findings, then the
aggregator merging them. "Each specialist is small enough to test and evaluate
on its facet — that's how we keep a big agent trustworthy." Tease: "but who
checks the specialists? A fact-checker, next."

## Next phase

`phase-23` — fact-checker + critic with a verification cycle.
