# Phase 19 — Routing, cycles & parallel execution

> **Git tag:** `phase-19`
> **Checkout:** `git checkout phase-19`
> **One-liner:** use LangGraph's real powers — parallel branches, a conditional
> edge, and a guarded cycle.

## Where we are

Phase 18's graph was linear — nothing a script couldn't do. Phase 19 shows why
the graph exists: two research branches run **in parallel**, a **conditional
edge** decides whether to gather more, and a **cycle** loops back — bounded by a
termination guard.

```text
                 ┌──> web_research ─────┐
START → plan ────┤                      ├──> analyze ──more?──> more_research ─┐
                 └──> internal_research ┘        │                            │
                                                 └──done──> synthesize → END   │
                        ▲──────────────────────────────────────────────────────┘
```

## Concepts explained

- **Parallel fan-out + join**: `plan` → `web_research` *and* `internal_research`
  concurrently; the `evidence` reducer merges both; `analyze` waits for both.
- **Conditional edges**: `add_conditional_edges(node, router, {...})` — a router
  function returns the next node's key.
- **Cycles**: `analyze → more_research → analyze` loops to deepen research.
- **Termination guard**: an iteration counter (`MAX_RESEARCH_LOOPS`) so the
  cycle always ends — the #1 way agents hang is an unguarded loop.
- **Streaming**: `graph.stream(...)` surfaces node-by-node execution — great for
  showing the parallel + cyclic order live.

## Files created / modified

```text
src/ai_research_agent/graph/advanced.py    # build_branching_graph + nodes/router
src/ai_research_agent/graph/state.py        # + analysis, iterations
src/ai_research_agent/graph/__init__.py
examples/09_langgraph/branching_graph.py
tests/agents/test_branching_graph.py
```

## Commands to run (on camera)

```bash
uv run pytest tests/agents/test_branching_graph.py -q
uv run python examples/09_langgraph/branching_graph.py
```

## Expected output

The ASCII graph, then a streamed node order:
`plan → web_research → internal_research → analyze → more_research → analyze →
synthesize` — parallel branches, then one loop of the cycle, then out.

## Demo moment

Point at the streamed order: web and internal ran together, then the router sent
us back through `analyze` once before `synthesize`. "The cycle deepened the
research; the guard guaranteed it stopped." Change `MIN_EVIDENCE` and rerun to
show more/fewer loops.

## Next phase

`phase-20` — persistence, checkpoints, interrupts, and human-in-the-loop.
