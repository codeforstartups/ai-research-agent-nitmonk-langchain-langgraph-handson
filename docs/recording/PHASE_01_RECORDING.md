# Phase 01 — LLM foundation

> **Git tag:** `phase-01`
> **Checkout:** `git checkout phase-01`
> **One-liner:** talk to a model through LangChain — provider-agnostically —
> with messages and streaming.

## Where we are

Phase 00 gave us a clean project. Now we make it *think*. We add a **single
factory**, `get_chat_model()`, that builds a chat model for Anthropic, OpenAI
or Gemini based on config — and every later phase asks for "a model" without
caring which provider is behind it.

## Concepts explained

- **Chat messages**: `SystemMessage` / `HumanMessage` / `AIMessage`, and why a
  chat model takes a *list* of messages, not a string.
- **Provider-agnostic construction**: the factory pattern that keeps the agent
  portable. Switch providers by changing one env var (`LLM_PROVIDER`).
- **`.invoke()` vs `.stream()`**: single response vs token streaming.
- **Graceful degradation**: `OfflineChatModel` — an honest stub so the whole
  course runs without API keys (CI, offline, pre-signup). Same code path; add a
  key and real intelligence flows through it.
- **`message.text`** (property, LangChain v1) for provider-neutral text access.

## Files created / modified

```text
src/ai_research_agent/models/__init__.py
src/ai_research_agent/models/chat.py        # get_chat_model() factory
src/ai_research_agent/models/offline.py     # OfflineChatModel fallback
src/ai_research_agent/cli.py                # + `ask` command (--stream)
examples/01_llm/basic_chat.py
tests/unit/test_chat_model.py
pyproject.toml                              # + langchain-core + 3 providers
```

## Commands to run (on camera)

```bash
uv sync                                   # pulls langchain + providers
uv run pytest -q
uv run python examples/01_llm/basic_chat.py

# The CLI now answers questions:
uv run research-agent ask "What is MCP in one line?"
uv run research-agent ask "Summarise LangGraph" --stream
```

## Expected output

Without keys you'll see a yellow `⚠ No key … using OfflineChatModel` banner,
then a deterministic stub answer (and a token-streamed version). With a key in
`.env`, the identical commands produce real model output.

## Demo moment

Run `ask` with no key (offline stub), then paste a real `ANTHROPIC_API_KEY`
into `.env` and rerun the *same command* — real answer, zero code change. Then
flip `LLM_PROVIDER=openai` and show it again. That's the pluggable-provider
payoff.

## Where this phase ends

We can call any of three providers through one interface, stream tokens, and
still run fully offline. Next we make the model return **typed, structured
data** instead of free text.

## Next phase

`phase-02` — structured outputs with Pydantic.
