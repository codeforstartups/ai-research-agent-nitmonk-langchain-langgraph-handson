# Phase 23 — Fact-checker & critic

> **Git tag:** `phase-23`
> **Checkout:** `git checkout phase-23`
> **One-liner:** gate the answer behind a grounding fact-check (with a
> verification cycle) and a critic.

## Where we are

Specialists can be confidently wrong. Phase 23 adds the reliability layer: a
**fact-check** that verifies the answer is grounded in cited evidence (looping
back if not), then a **critic** that sharpens the grounded result.

```text
… → aggregate → fact_check ─fail→ revise → aggregate (cycle, capped)
                    └─pass→ critic → END
```

## Concepts explained

- **Fact-check as a gate**: a typed `Verdict{passed, issues, note}` from a
  grounding check (answer non-trivial + cites known sources).
- **Verification cycle**: on failure, loop back to gather/aggregate again;
  `MAX_VERIFY_LOOPS` guarantees termination.
- **Critic**: a qualitative pass that improves clarity and forces an explicit
  recommendation — without adding unsupported claims.
- **Layered reliability**: retrieve → specialise → aggregate → verify → critique.

## Files created / modified

```text
src/ai_research_agent/graph/verification.py    # fact_check, route, revise, critic
examples/09_langgraph/verified_graph.py
tests/agents/test_verification.py
```

## Commands to run (on camera)

```bash
uv run pytest tests/agents/test_verification.py -q
uv run python examples/09_langgraph/verified_graph.py
```

## Expected output

The node order runs supervisor → 3 specialists → aggregate → fact_check → critic,
with `Verdict passed: True` (the answer is grounded), then a sharpened final
answer.

## Demo moment

Force a failure — temporarily blank the sources — and show the graph looping back
through `revise` before the cap sends it on. "The fact-check is what turns a
plausible answer into a *defensible* one."

## Next phase

`phase-24` — Deep Agents: the higher-level harness.
