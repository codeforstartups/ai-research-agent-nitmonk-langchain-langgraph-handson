# Phase 02 — Structured output

> **Git tag:** `phase-02`
> **Checkout:** `git checkout phase-02`
> **One-liner:** make the model return a *typed Pydantic object*, not a string.

## Where we are

We can talk to a model. But free text is fragile — an agent that passes strings
between steps and re-parses them constantly is a bug factory. Phase 02 makes the
model return **validated, typed data**. The first artifact: a `ResearchPlan`.

## Concepts explained

- **`.with_structured_output(Schema)`** — bind a Pydantic schema so the provider
  returns JSON that LangChain validates into a real object.
- **Why typed I/O matters**: validation at the boundary, safe hand-off between
  graph nodes, self-documenting contracts.
- **Schema design**: `SubQuestion` / `ResearchPlan` with `Field(description=...)`
  — those descriptions become part of the instruction to the model.
- **Offline-safe structured output**: a heuristic offline builder for known
  schemas + a generic placeholder builder for any other schema.

## Files created / modified

```text
src/ai_research_agent/schemas/__init__.py
src/ai_research_agent/schemas/research.py       # SubQuestion, ResearchPlan
src/ai_research_agent/models/structured.py      # get_structured_response()
src/ai_research_agent/models/__init__.py        # export it
src/ai_research_agent/cli.py                     # + `plan` command
examples/01_llm/structured_output.py
tests/unit/test_structured_output.py
```

## Commands to run (on camera)

```bash
uv run pytest -q
uv run python examples/01_llm/structured_output.py
uv run research-agent plan "Should we build an enterprise AI coding agent?"
```

## Expected output

A `ResearchPlan` printed as an objective plus 3 tagged sub-questions
(`(web)` / `(internal)`). Offline it's the deterministic heuristic plan; with a
key it's a real, model-generated decomposition — same `get_structured_response`
call.

## Demo moment

Show the `type: ResearchPlan • validated by Pydantic` line, then open
`schemas/research.py` and point out that the `Field(description=...)` text is
what steers the model. Change a description, rerun with a real key, watch the
output shift.

## Where this phase ends

The agent can now emit typed plans. Next we leave pure-LLM territory and start
building **retrieval** — beginning with embeddings.

## Next phase

`phase-03` — embeddings & semantic similarity.
