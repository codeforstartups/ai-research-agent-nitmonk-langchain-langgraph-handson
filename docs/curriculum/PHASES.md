# Phase Roadmap

This repository is built **phase by phase**. Every phase is one git commit and
one git tag (`phase-XX`). Check out any tag to land on a complete, runnable
teaching checkpoint:

```bash
git checkout phase-05      # jump to "vector retrieval"
git checkout main          # back to the latest state
git tag                    # list every phase
```

Each phase also has a recording guide in [`docs/recording/`](../recording/)
(`PHASE_XX_RECORDING.md`) with the exact commands, expected output, concepts,
and the "demo moment" to capture on camera.

The **BUILD → RUN → TRACE → EVALUATE → IMPROVE** loop is the spine of the
course: we don't just add frameworks, we add them to solve a problem the
previous phase exposed.

| Tag | Title | Introduces | Key libraries added |
|-----|-------|-----------|---------------------|
| `phase-00` | Bootstrap | uv project, pluggable config, CLI `doctor` | pydantic-settings, typer, rich |
| `phase-01` | LLM foundation | pluggable chat models, messages, streaming | langchain, langchain-{anthropic,openai,google-genai} |
| `phase-02` | Structured output | Pydantic-typed model responses | (uses phase-01 deps) |
| `phase-03` | Embeddings | pluggable embeddings, similarity | fastembed |
| `phase-04` | Ingestion | loaders, chunking, metadata | langchain-text-splitters, pypdf |
| `phase-05` | Vector retrieval | Pinecone vector store + retriever (in-memory double for offline) | langchain-pinecone, pinecone |
| `phase-06` | RAG pipeline | LCEL retrieve→prompt→generate | (uses existing deps) |
| `phase-07` | BM25 | keyword/sparse retrieval | rank-bm25 |
| `phase-08` | Hybrid search | reciprocal-rank fusion | (uses existing deps) |
| `phase-09` | Reranking | cross-encoder reranker | flashrank |
| `phase-10` | Ragas eval | faithfulness / relevance / context metrics | ragas, datasets |
| `phase-11` | LangChain tools | `@tool`, schemas, tool calling | (uses existing deps) |
| `phase-12` | MCP foundation | MCP server, tool, resource, prompt, Inspector | mcp |
| `phase-13` | Web Research MCP | Thordata-backed web tools | httpx, tenacity |
| `phase-14` | Knowledge MCP | RAG exposed as MCP server | (uses existing deps) |
| `phase-15` | Multi-MCP client | connect many servers, dynamic discovery | (uses existing deps) |
| `phase-16` | MCP + LangChain | adapt MCP tools into LangChain tools | langchain-mcp-adapters |
| `phase-17` | LangChain agent | tool-calling agent loop | (uses existing deps) |
| `phase-18` | LangGraph foundation | State, nodes, edges | langgraph |
| `phase-19` | Routing & cycles | conditional edges, parallel fan-out | (uses existing deps) |
| `phase-20` | Persistence & HITL | checkpoints, interrupts, human review | langgraph-checkpoint-sqlite |
| `phase-21` | MCP + LangGraph | graph nodes call MCP capabilities | (uses existing deps) |
| `phase-22` | Subagents | supervisor + specialist researchers | (uses existing deps) |
| `phase-23` | Fact-check & critic | verification cycle, critic gate | (uses existing deps) |
| `phase-24` | Deep Agents | planning, skills, subagents, filesystem | deepagents |
| `phase-25` | Observability | Langfuse traces across the whole run | langfuse |
| `phase-26` | Final assembly | full graph + `research-agent run` CLI | (uses existing deps) |
| `phase-27` | Final demo & eval | end-to-end demo, retrieval benchmark | (uses existing deps) |

> The numbering may evolve, but every major teaching milestone stays
> recoverable via its tag. When a phase's design decision is non-obvious it is
> also recorded in [`docs/decisions/`](../decisions/) as an ADR.

## The final architecture (target)

```text
USER ─▶ PLANNER ─▶ ┌── WEB RESEARCH (MCP ▶ Thordata)
                   └── INTERNAL KNOWLEDGE (MCP ▶ Hybrid RAG ▶ Reranker)
                          │
                          ▼
                       ANALYST ─▶ FACT CHECKER ─▶ (fail ▶ more research)
                          │                         (pass ▼)
                          ▼                        CRITIC ─▶ HUMAN REVIEW ─▶ REPORT
```

Everything above is traced in Langfuse and the retrieval stack is measured with
Ragas.
