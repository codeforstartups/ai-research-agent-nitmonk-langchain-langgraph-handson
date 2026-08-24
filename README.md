# AI Research & Analysis Agent

A **teaching repository** that progressively engineers a reliable AI research
agent on the modern LangChain / LangGraph / MCP / Deep Agents stack — built as
the companion source code for a long-form engineering video.

You hand the agent a hard question:

> *"Analyze the AI coding-agent market and determine whether an enterprise
> startup should enter it."*

…and it plans, researches the live web (via **Thordata** through **MCP**),
consults an internal knowledge base (**hybrid RAG + reranking**), analyses and
**fact-checks** its findings in a **LangGraph** cycle with **human review**, all
**traced in Langfuse** and **evaluated with Ragas**.

## Why this repo is different

It is simultaneously a working agent *and* a line-by-line course. It is built
in ~28 numbered **phases**, each a single git commit and tag. The philosophy is
a loop, not a feature checklist:

```text
BUILD → RUN → TRACE → EVALUATE → IDENTIFY PROBLEM → IMPROVE → REPEAT
```

Frameworks earn their place by solving a problem the previous phase exposed —
never "look how many tools I can import".

| Layer | Role |
|-------|------|
| **LangChain** | building blocks: models, prompts, structured output, embeddings, retrievers, tools |
| **LangGraph** | stateful orchestration: nodes, edges, cycles, persistence, HITL |
| **MCP** | standardized external-capability boundary (multiple servers) |
| **Deep Agents** | higher-level agent harness: planning, skills, subagents |
| **Langfuse** | observability across the whole run |
| **Ragas** | retrieval + RAG evaluation |
| **Thordata** | live web-data acquisition |

## Navigating the phases

```bash
git tag                      # list all phases
git checkout phase-12        # jump to "MCP foundation"
cat docs/recording/PHASE_12_RECORDING.md
git checkout main            # latest
```

The full map is in [`docs/curriculum/PHASES.md`](docs/curriculum/PHASES.md).
Every phase has a recording guide in [`docs/recording/`](docs/recording/).

## Quick start

```bash
# 1. Install uv (https://astral.sh/uv) if you don't have it, then:
uv sync --extra dev

# 2. Configure (optional — everything runs with local fallbacks until keys exist)
cp .env.example .env         # add ANTHROPIC_API_KEY / OPENAI_API_KEY / ... as you get them

# 3. Verify the environment
uv run research-agent doctor
```

`doctor` prints the resolved configuration and shows which live services are
wired up. Green = ready, yellow = using a local fallback.

## Project layout

```text
src/ai_research_agent/   # the agent (config, models, retrieval, mcp, graph, ...)
examples/                # runnable per-concept demos (01_llm ... 11_evaluation)
tests/                   # unit / integration / mcp / retrieval / agents / evaluation
evals/                   # Ragas datasets, experiments, reports
docs/                    # curriculum, architecture, concepts, decisions, recording
data/                    # raw / processed / vectorstore (git-ignored contents)
```

## Requirements

* Python ≥ 3.10
* [uv](https://astral.sh/uv) for environment + dependency management
* API keys are all optional to start; add them to `.env` to enable live services.

---

*Built phase by phase. Start at [`phase-00`](docs/recording/PHASE_00_RECORDING.md).*
