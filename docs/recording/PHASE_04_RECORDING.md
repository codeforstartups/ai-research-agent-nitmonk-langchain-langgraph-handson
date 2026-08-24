# Phase 04 — Document ingestion & chunking

> **Git tag:** `phase-04`
> **Checkout:** `git checkout phase-04`
> **One-liner:** load a knowledge base and split it into embeddable, traceable
> chunks.

## Where we are

We can embed and rank text. Now we need *text to embed*. Phase 04 builds the
front of the RAG pipeline: load heterogeneous files into a common `Document`
type and chunk them, carrying metadata for citations.

## Concepts explained

- **`Document`** = `page_content` + `metadata`; the universal unit of RAG.
- **Loaders**: markdown/text read directly, PDFs via `pypdf` (one doc per page).
- **Why chunk**: embeddings blur long text; prompts have limits; chunks are
  coherent, retrievable units.
- **`RecursiveCharacterTextSplitter`**: split on natural boundaries first
  (paragraph → line → word), with **overlap** so cross-boundary ideas survive.
- **Metadata for trust**: every chunk keeps `source` + `chunk` index → citations
  and the untrusted-content boundary we enforce later.

## Files created / modified

```text
data/knowledge/*.md                              # illustrative sample corpus
src/ai_research_agent/ingestion/loaders.py
src/ai_research_agent/ingestion/chunking.py
src/ai_research_agent/ingestion/__init__.py
src/ai_research_agent/cli.py                       # + `ingest` command
src/ai_research_agent/config/settings.py           # knowledge_dir -> data/knowledge
examples/04_rag/ingest_and_chunk.py
tests/retrieval/test_ingestion.py
pyproject.toml                                     # + langchain-text-splitters, pypdf
```

## Commands to run (on camera)

```bash
uv run pytest -q
uv run python examples/04_rag/ingest_and_chunk.py
uv run research-agent ingest
```

## Expected output

`ingest` loads 4 sample documents from `data/knowledge/` and reports 9 chunks,
broken down per source. The example prints the first 3 chunks with their
`source#chunk` tags and character counts.

## Demo moment

Open `chunking.py`, change `chunk_size` from 800 to 300, rerun `ingest`, and
watch the chunk count jump — a tangible way to explain the size/precision
trade-off before we index these chunks into a vector store next.

## Next phase

`phase-05` — vector retrieval with a Pinecone vector store.
