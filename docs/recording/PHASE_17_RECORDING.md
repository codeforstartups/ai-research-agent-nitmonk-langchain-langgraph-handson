# Phase 17 — LangChain tool-calling agent

> **Git tag:** `phase-17`
> **Checkout:** `git checkout phase-17`
> **One-liner:** wrap the tool-calling loop into an agent — by hand, so
> LangGraph's value is obvious next.

## Where we are

We can bind tools and run one turn. Phase 17 closes the loop into an **agent**:
model → tools → model → … → answer, drawing tools from every MCP server. We
write it explicitly on purpose — phase 18 replaces this `for` loop with a
LangGraph state machine, and you'll see exactly what you gain.

## Concepts explained

- **The agent loop**: request tools → execute → feed `ToolMessage`s back →
  repeat until the model answers.
- **Safety rails**: an iteration cap (no infinite loops); tool errors captured,
  not crashed on.
- **Tools from everywhere**: the agent's toolset is the MCP tool surface via the
  adapter — add a server, the agent can use it.
- **Honest offline behaviour**: no key → no tool calls → the stub answer. "No
  key, no autonomy" is a real, visible property.

## Files created / modified

```text
src/ai_research_agent/agents/tool_agent.py    # run_tool_agent + AgentResult/Step
src/ai_research_agent/agents/__init__.py
src/ai_research_agent/models/offline.py        # bind_tools (accept + ignore)
examples/08_agents/tool_agent.py
tests/agents/test_tool_agent.py                # scripted-model loop tests
```

## Commands to run (on camera)

```bash
uv run pytest tests/agents -q
uv run python examples/08_agents/tool_agent.py     # needs a key to call tools
```

## Expected output

Tests drive a scripted model: it requests `calculate(6*7)`, we run it (`42.0`),
it answers "The answer is 42." — proving the loop executes tools and terminates.
With a real key, the example shows the agent calling `knowledge__search_knowledge`
and answering from the retrieved evidence.

## Demo moment

Walk the `run_tool_agent` loop line by line, then say: "this works — but where's
the *state*? Retries? Human approval? Resuming after a crash? Bolting those onto
a `for` loop gets ugly fast. That's why we move to LangGraph."

## Next phase

`phase-18` — LangGraph foundation: State, nodes, edges.
