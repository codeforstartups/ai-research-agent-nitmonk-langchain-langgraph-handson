# Phase 18 — LangGraph foundation

> **Git tag:** `phase-18`
> **Checkout:** `git checkout phase-18`
> **One-liner:** rebuild the research flow as a LangGraph state machine —
> State, nodes, edges.

## Where we are

Phase 17's agent was a `for` loop. Phase 18 makes the flow a **graph**: a typed
State threaded through nodes wired by edges. Same work, but now inspectable,
visualizable, and ready for routing / cycles / persistence / human review.

```text
START → plan → research → synthesize → END
```

## Concepts explained

- **State** (`ResearchState`): a `TypedDict`; nodes return *partial* updates that
  LangGraph merges. **Reducers** (`Annotated[list, operator.add]`) accumulate
  instead of overwrite — the key to parallel branches later.
- **Nodes** are plain functions `state → partial state` — unit-testable alone.
- **Edges** wire the flow; `START` / `END` are the entry/exit sentinels.
- **Compile & run**: `graph.compile().invoke({"question": ...})`.
- **Visualization**: `graph.get_graph().draw_ascii()` (via grandalf).

## Files created / modified

```text
src/ai_research_agent/graph/state.py     # ResearchState + reducers
src/ai_research_agent/graph/nodes.py     # plan / research / synthesize
src/ai_research_agent/graph/build.py     # build_research_graph / run_research
src/ai_research_agent/graph/__init__.py
examples/09_langgraph/linear_graph.py
tests/agents/test_graph.py
pyproject.toml                            # + langgraph, grandalf
```

## Commands to run (on camera)

```bash
uv run pytest tests/agents/test_graph.py -q
uv run python examples/09_langgraph/linear_graph.py
```

## Expected output

An ASCII diagram of `__start__ → plan → research → synthesize → __end__`, then a
run: a 3-sub-question plan, ~8 evidence chunks, and a synthesized answer.

## Demo moment

Show the ASCII graph, then open `nodes.py`: "these are the *same* operations as
the agent loop — planning, retrieving, synthesizing — but now each is a node in
a machine we can reshape. Watch what that buys us." Then tease routing + cycles.

## Next phase

`phase-19` — conditional edges, cycles, and parallel fan-out.
