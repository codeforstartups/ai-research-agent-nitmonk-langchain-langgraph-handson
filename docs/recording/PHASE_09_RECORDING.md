# Phase 09 — Reranking (cross-encoder)

> **Git tag:** `phase-09`
> **Checkout:** `git checkout phase-09`
> **One-liner:** add a cross-encoder reranking stage for precise top-k results.

## Where we are

Retrieval scores query and document independently — fast, but coarse. A
**cross-encoder** reads them *together* and judges true relevance. We use it as
a second stage: retrieve many cheaply, then rerank the shortlist precisely.

## Concepts explained

- **Bi-encoder vs cross-encoder**: independent vs joint scoring; accuracy vs cost.
- **Retrieve-then-rerank**: wide net (`hybrid k=6`) → precise `top_n=3`.
- **FlashRank**: small local ONNX cross-encoders (auto-downloads a tiny model).
- **Announced degradation**: offline it falls back to a lexical reranker with a
  visible warning — never a silent quality drop.
- **Same interface**: `RerankingRetriever` is a `BaseRetriever`, so it drops
  into RAG/agent unchanged.

## Files created / modified

```text
src/ai_research_agent/reranking/reranker.py     # Reranker (FlashRank + fallback)
src/ai_research_agent/reranking/retriever.py    # RerankingRetriever
src/ai_research_agent/reranking/__init__.py
examples/05_hybrid_search/reranking.py
tests/retrieval/test_reranking.py
pyproject.toml                                   # + flashrank
```

## Commands to run (on camera)

```bash
uv run pytest -q
uv run python examples/05_hybrid_search/reranking.py
```

## Expected output

The hybrid candidate list (6) is reordered into a top-3 where the passage most
directly answering the governance query rises to #1 (`enterprise_requirements`).

## Demo moment

Show a candidate that was mid-pack after hybrid retrieval jumping to #1 after
reranking. Then set up the payoff: "We now have four retrieval configurations —
vector, BM25, hybrid, hybrid+rerank. Which is best? Next phase we *measure* it
with Ragas instead of eyeballing."

## Next phase

`phase-10` — Ragas evaluation of the retrieval stack.
