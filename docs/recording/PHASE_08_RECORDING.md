# Phase 08 — Hybrid search (RRF)

> **Git tag:** `phase-08`
> **Checkout:** `git checkout phase-08`
> **One-liner:** fuse BM25 and vector rankings into one robust list with
> reciprocal-rank fusion.

## Where we are

We have two complementary retrievers. Phase 08 merges them. **RRF** fuses ranked
lists using only ranks (not scores), so incompatible score scales don't matter:
`score(d) = Σ 1 / (rrf_k + rank_d)` across retrievers.

## Concepts explained

- **Reciprocal Rank Fusion**: rank-based, scale-free list merging.
- **Why it's a great default**: hits from *either* retriever surface; hits from
  *both* get boosted; de-duplication by stable doc key.
- **Composability**: `HybridRetriever` is itself a `BaseRetriever`, so RAG and
  the agent consume it identically to any single retriever.
- **Wide-net then fuse**: each sub-retriever fetches more candidates
  (`per_retriever_k`) than the final `k`.

## Files created / modified

```text
src/ai_research_agent/retrieval/hybrid.py      # RRF + HybridRetriever + builder
src/ai_research_agent/retrieval/__init__.py
examples/05_hybrid_search/hybrid_fusion.py
tests/retrieval/test_hybrid.py
```

## Commands to run (on camera)

```bash
uv run pytest -q
uv run python examples/05_hybrid_search/hybrid_fusion.py
```

## Expected output

Three lists — BM25-only, vector-only, hybrid. The hybrid list pulls the top hit
from *each* of the other two to its front, demonstrating fusion.

## Demo moment

Show a document that was #1 for vectors but low for BM25 (or vice-versa) landing
near the top of the hybrid list. "One knob — `k` — now backed by two retrieval
strategies. But is hybrid *actually* better? We stop guessing and measure it
with Ragas — but first, one more quality lever: reranking."

## Next phase

`phase-09` — reranking with a cross-encoder.
