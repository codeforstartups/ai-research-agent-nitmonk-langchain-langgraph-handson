# Phase 00 — Bootstrap the uv project

> **Git tag:** `phase-00`
> **Checkout:** `git checkout phase-00`
> **One-liner:** stand up a clean, pluggable, well-configured Python project so
> every later phase has a solid foundation.

## Where we are

The very beginning. Empty directory → a real, installable Python package with
typed configuration and a working CLI. Nothing "AI" yet — this phase is about
doing the boring parts *well* so the interesting parts stay clean.

## Concepts explained

- **uv** as the project/dependency manager (`uv sync`, `uv run`, `uv add`).
- **`src/` layout** and why it prevents import surprises.
- **Typed configuration** with `pydantic-settings` — validate config at startup.
- **Pluggable providers** — the `LLMProvider` / `EmbeddingsProvider` enums that
  later let the same agent run on Anthropic, OpenAI or Gemini.
- **Per-phase dependency growth** — the pedagogical reason we don't install
  everything at once.

## Files created

```text
pyproject.toml                          # uv project + phase-00 deps
.python-version                         # pins 3.10
.gitignore  .env.example
src/ai_research_agent/__init__.py
src/ai_research_agent/config/           # settings.py — pluggable, typed config
src/ai_research_agent/utils/            # logging.py — shared Rich console
src/ai_research_agent/cli.py            # `research-agent` CLI (doctor, version)
tests/unit/test_config.py
docs/curriculum/PHASES.md               # the roadmap
README.md  CLAUDE.md
```

## Commands to run (on camera)

```bash
# Install the environment
uv sync --extra dev

# Run the tests — configuration loads and validates
uv run pytest

# Lint
uv run ruff check .

# The demo moment: environment doctor
uv run research-agent doctor
```

## Expected output

`research-agent doctor` prints two tables — resolved **Configuration** and
**Live services** — followed by a green `Phase 00 ✓` panel. With no `.env`,
every live service shows `○ not configured (using fallback)`; that's correct.

`uv run pytest` shows **4 passed**.

## Demo moment

Run `uv run research-agent doctor`, then create `.env` from `.env.example`, drop
in one key (e.g. `ANTHROPIC_API_KEY`), rerun, and watch Anthropic flip to
`● ready`. That visual — config driving behaviour — is the through-line of the
whole course.

## Where this phase ends

A clean, green checkpoint: installable package, passing tests, working CLI,
roadmap documented. Next we make it actually talk to a model.

## Next phase

`phase-01` — LLM foundation: pluggable chat models, messages, and streaming.
