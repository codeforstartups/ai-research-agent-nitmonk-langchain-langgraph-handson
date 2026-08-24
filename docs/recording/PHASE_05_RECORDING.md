# Phase 05 — Vector retrieval (Pinecone)

> **Git tag:** `phase-05`
> **Checkout:** `git checkout phase-05`
> **One-liner:** persist chunk embeddings in a vector database and retrieve by
> similarity.

> **Note:** the course uses **Pinecone** (serverless) as the product vector
> store. Pinecone is cloud-hosted with no local mode, so the CLI path
> (`ingest`/`search`/`rag`) needs `PINECONE_API_KEY`. The teaching examples and
> tests use langchain-core's in-memory `InMemoryVectorStore`
> (`build_ephemeral_retriever`) so the concepts run with no key.

## Where we are

Phase 03's similarity loop was O(n) and forgot everything between runs. Now we
put the chunks from phase 04 into **Pinecone**, a persistent serverless vector
database, and retrieve through LangChain's `Retriever` interface.

## Concepts explained

- **Vector database**: stores + indexes embeddings for fast nearest-neighbour
  search and persists in the cloud; the index is sized to your embedding's
  dimension and created on first `ingest`.
- **Pluggable embeddings into Pinecone**: the store is handed our
  `get_embeddings()` model, so it works with fastembed/OpenAI/Google.
- **`Retriever`**: `store.as_retriever(...).invoke(query)` → `Document`s. This
  is the interface every later component (RAG, agents, graph) consumes.
- **Serverless index lifecycle**: `ensure_index()` creates the Pinecone index
  (sized to the embedding dimension) on first use and waits until it's ready.
- **Product store vs teaching double**: Pinecone for the persistent product path
  (`ingest`/`search`), `build_ephemeral_retriever()` (in-memory) for offline
  examples and tests — same `Retriever` interface.

## Files created / modified

```text
src/ai_research_agent/retrieval/vectorstore.py    # build/load/retrieve/index
src/ai_research_agent/retrieval/__init__.py
src/ai_research_agent/cli.py                        # ingest now indexes; + search
examples/04_rag/vector_search.py
tests/retrieval/test_vectorstore.py
pyproject.toml                                      # + langchain-pinecone, pinecone
```

## Commands to run (on camera)

```bash
uv run pytest -q
uv run python examples/04_rag/vector_search.py       # in-memory, no key
# With PINECONE_API_KEY set in .env:
uv run research-agent ingest                         # creates + fills the index
uv run research-agent search "enterprise requirements for AI agents" --k 3
```

## Expected output

The example prints top-3 in-memory results. With a Pinecone key, `ingest`
reports `4 documents → 9 chunks` indexed and `search` returns the top-k chunks,
each with a `source#chunk` citation, ordered by semantic relevance.

## Demo moment

Run `search` for a compliance question and show it surfacing
`enterprise_requirements.md`. Then note: we're returning *chunks*, not an
*answer*. Next phase we feed these chunks to the model — that's RAG.

## Next phase

`phase-06` — the RAG pipeline (LCEL: retrieve → prompt → generate).
