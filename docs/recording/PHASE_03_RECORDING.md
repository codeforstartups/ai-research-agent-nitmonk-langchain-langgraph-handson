# Phase 03 — Embeddings & semantic similarity

> **Git tag:** `phase-03`
> **Checkout:** `git checkout phase-03`
> **One-liner:** turn text into vectors and rank by meaning with cosine
> similarity — the foundation of RAG.

## Where we are

We leave pure-LLM territory. To let the agent consult a knowledge base, we first
need to represent text as **vectors** and measure **semantic closeness**. No
vector DB yet — just the raw idea, done by hand, so the machinery is visible.

## Concepts explained

- **Embeddings**: text → a point in high-dimensional meaning-space. The
  `Embeddings` interface: `embed_documents`, `embed_query`.
- **Pluggable backends**: local **fastembed** (ONNX, no key — the default),
  OpenAI, Google — behind one `get_embeddings()`.
- **Cosine similarity**: angle between vectors; the ranking signal for search.
- **Semantic vs keyword**: the query "companies building autonomous coding
  agents" ranks Devin/Cursor/Copilot high without sharing exact words.
- **Deterministic fallback**: a hashing embedding for offline/CI so tests never
  hit the network.

## Files created / modified

```text
src/ai_research_agent/embeddings/__init__.py
src/ai_research_agent/embeddings/factory.py          # get_embeddings()
src/ai_research_agent/embeddings/fastembed_local.py   # local ONNX wrapper
src/ai_research_agent/embeddings/deterministic.py     # offline fallback
src/ai_research_agent/embeddings/similarity.py        # cosine + ranking
examples/02_embeddings/basic_embeddings.py
examples/03_similarity/semantic_search.py
tests/retrieval/test_embeddings.py
pyproject.toml                                        # + fastembed, numpy
```

## Commands to run (on camera)

```bash
uv run pytest -q
uv run python examples/02_embeddings/basic_embeddings.py     # what a vector is
uv run python examples/03_similarity/semantic_search.py      # rank by meaning
```

> First run downloads `BAAI/bge-small-en-v1.5` (~small) and caches it.

## Expected output

`semantic_search.py` ranks the AI-coding-agent documents at the top (~0.6–0.7
cosine) and unrelated ones (Paris, photosynthesis) at the bottom (~0.3) — proof
that similarity is capturing meaning, not keywords.

## Demo moment

Change `EMBEDDINGS_PROVIDER=openai` in `.env` (with a key) and rerun — same code,
different backend, still ranks correctly. Then point out the O(n) loop and say:
"this is what a vector database will do for us at scale — next."

## Next phase

`phase-04` — document ingestion & chunking.
