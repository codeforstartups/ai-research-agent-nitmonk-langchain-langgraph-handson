# Phase 05 — Vector retrieval (Chroma)

> **Git tag:** `phase-05`
> **Checkout:** `git checkout phase-05`
> **One-liner:** persist chunk embeddings in a vector database and retrieve by
> similarity.

## Where we are

Phase 03's similarity loop was O(n) and forgot everything between runs. Now we
put the chunks from phase 04 into **Chroma**, a persistent vector database, and
retrieve through LangChain's `Retriever` interface.

## Concepts explained

- **Vector database**: stores + indexes embeddings for fast nearest-neighbour
  search and persists to disk.
- **Pluggable embeddings into Chroma**: the store is handed our
  `get_embeddings()` model, so it works with fastembed/OpenAI/Google.
- **`Retriever`**: `store.as_retriever(...).invoke(query)` → `Document`s. This
  is the interface every later component (RAG, agents, graph) consumes.
- **Idempotent ingest**: `reset_collection()` so re-running `ingest` rebuilds
  rather than duplicating.
- **Ephemeral vs persistent**: `persist_directory=None` for tests, a path for
  the real index.

## Files created / modified

```text
src/ai_research_agent/retrieval/vectorstore.py    # build/load/retrieve/index
src/ai_research_agent/retrieval/__init__.py
src/ai_research_agent/cli.py                        # ingest now indexes; + search
examples/04_rag/vector_search.py
tests/retrieval/test_vectorstore.py
pyproject.toml                                      # + langchain-chroma, chromadb
```

## Commands to run (on camera)

```bash
uv run pytest -q
uv run python examples/04_rag/vector_search.py
uv run research-agent ingest                         # writes data/vectorstore/
uv run research-agent search "enterprise requirements for AI agents" --k 3
```

## Expected output

`ingest` reports `4 documents → 9 chunks` indexed. `search` returns the top-k
chunks, each with a `source#chunk` citation, ordered by semantic relevance.

## Demo moment

Run `search` for a compliance question and show it surfacing
`enterprise_requirements.md`. Then note: we're returning *chunks*, not an
*answer*. Next phase we feed these chunks to the model — that's RAG.

## Next phase

`phase-06` — the RAG pipeline (LCEL: retrieve → prompt → generate).
