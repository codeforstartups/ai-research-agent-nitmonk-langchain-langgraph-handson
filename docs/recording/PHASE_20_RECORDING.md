# Phase 20 — Persistence, checkpoints & human-in-the-loop

> **Git tag:** `phase-20`
> **Checkout:** `git checkout phase-20`
> **One-liner:** save state at every step and pause for a human to approve the
> answer.

## Where we are

This is the payoff for using a graph. Attach a **checkpointer** and the run gains
memory: it can pause, be inspected, and resume — including waiting for a
**human** to approve or reject the draft before it's finalised.

```text
plan → research → draft → human_review ─approved→ END
                             │
                             └─rejected→ revise → human_review (loop, capped)
```

## Concepts explained

- **Checkpointer**: `compile(checkpointer=...)` saves state per `thread_id`
  (`InMemorySaver` for tests, `SqliteSaver` for durable on-disk persistence).
- **Interrupts**: `interrupt(payload)` suspends the graph and hands control back;
  `Command(resume=decision)` continues from the exact checkpoint.
- **Inspecting a paused run**: `graph.get_state(config).next` shows where it's
  paused; `.values` shows the saved state.
- **Human-in-the-loop**: approve → END; reject with feedback → `revise` → review
  again, bounded by `MAX_REVISIONS`.
- **Checkpoint security**: LangGraph won't deserialise arbitrary modules; we
  explicitly allow our own `ResearchPlan` schema.

## Files created / modified

```text
src/ai_research_agent/graph/hitl.py    # build_hitl_graph, interrupt, review router
src/ai_research_agent/graph/state.py    # + approved, human_feedback
examples/09_langgraph/human_in_the_loop.py
tests/agents/test_hitl.py
pyproject.toml                          # + langgraph-checkpoint-sqlite
```

## Commands to run (on camera)

```bash
uv run pytest tests/agents/test_hitl.py -q
uv run python examples/09_langgraph/human_in_the_loop.py
```

## Expected output

The graph runs to `⏸ Paused for human review`, prints the draft and the paused
node (`human_review`), then — on `Command(resume={"approved": True})` — finishes
with `Approved: True` and an empty `next` (done).

## Demo moment

Run to the interrupt, call `get_state()` to show the *saved* draft mid-run, then
resume with approval. Then reject with feedback and show it looping back through
`revise`. "The run survived a pause and a human decision — because the whole
state is checkpointed."

## Next phase

`phase-21` — let graph nodes call MCP capabilities.
