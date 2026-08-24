# Phase 07 — BM25 keyword retrieval

> **Git tag:** `phase-07`
> **Checkout:** `git checkout phase-07`
> **One-liner:** add sparse keyword retrieval to complement dense vectors.

## Where we are

Vector search matches meaning but can miss exact tokens — product names,
acronyms like `SOC2`, error codes. Phase 07 adds **BM25**, the classic keyword
ranking function, behind the same `Retriever` interface, setting up hybrid
search next.

## Concepts explained

- **Sparse vs dense**: BM25 scores by term overlap (rare terms weighted higher,
  repetition saturated) vs embeddings' semantic proximity.
- **When each wins**: BM25 for exact/rare terms; vectors for paraphrase.
- **`BaseRetriever`**: implementing `_get_relevant_documents` so BM25 is
  interchangeable with the vector retriever (`.invoke(query)`).
- **Zero-overlap handling**: a query sharing no vocabulary returns nothing
  rather than noise.

## Files created / modified

```text
src/ai_research_agent/retrieval/bm25.py        # BM25Retriever + tokenize
src/ai_research_agent/retrieval/__init__.py
examples/05_hybrid_search/bm25_vs_vector.py
tests/retrieval/test_bm25.py
pyproject.toml                                  # + rank-bm25
```

## Commands to run (on camera)

```bash
uv run pytest -q
uv run python examples/05_hybrid_search/bm25_vs_vector.py
```

## Expected output

Side-by-side results for the query `SOC2 compliance requirements`: BM25 locks
onto the chunk literally containing `SOC2`; the vector retriever brings
semantically-related passages. Neither list is a strict superset of the other —
that gap is the argument for fusion.

## Demo moment

Point at a result BM25 finds that vectors miss (the exact `SOC2` chunk) and one
vectors find that BM25 misses (a paraphrase with no shared words). "We want
both lists — merged intelligently. That's phase 08."

## Next phase

`phase-08` — hybrid search via reciprocal-rank fusion.
