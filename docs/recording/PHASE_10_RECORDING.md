# Phase 10 — Ragas evaluation & the retrieval benchmark

> **Git tag:** `phase-10`
> **Checkout:** `git checkout phase-10`
> **One-liner:** stop eyeballing retrieval quality — measure it, and judge
> answer quality with Ragas.

## Where we are

We built four retrieval configurations. Phase 10 closes the
**BUILD → RUN → EVALUATE** loop with two kinds of measurement:

1. an **offline retrieval benchmark** (hit@k / recall@k / MRR against a labeled
   set) — no keys needed;
2. **Ragas** answer-quality metrics (faithfulness, relevancy, context
   precision/recall) — needs an LLM judge.

## Concepts explained

- **IR metrics**: hit@k, recall@k, MRR — what they mean and when each matters.
- **Honest evaluation**: we report what we measure. On this small, clean corpus
  the result is instructive rather than triumphant (see below).
- **Ragas**: LLM-as-judge for generation quality; wrapping our pluggable
  model+embeddings with `LangchainLLMWrapper` / `LangchainEmbeddingsWrapper`.
- **Dependency reality**: a documented compat shim (`evaluation/_compat.py`)
  lets ragas 0.4 import under langchain-core v1 — a real-world integration
  wrinkle worth showing.

## The honest result (and why it's the point)

```
config           hit@k  recall@k   MRR
vector            1.00     1.00    0.94
bm25              1.00     1.00    0.87
hybrid            1.00     1.00    0.94
hybrid+rerank     1.00     1.00    0.86
```

On a *small, clean* corpus, dense vector retrieval is already near-perfect;
**hybrid matches it** (its value is robustness — it equals the best single
retriever without knowing in advance which that is); **BM25** trails on
paraphrased queries; and **reranking does not help a tiny candidate pool** — it
earns its keep when you have *many* noisy candidates to sort, not six clean ones.

> This is exactly the course philosophy: **measure before adding complexity, and
> don't force a component just because you built it.** At scale (large corpora,
> big candidate sets) the ranking flips toward hybrid+rerank — swap in your own
> corpus and re-run to see it.

## Files created / modified

```text
evals/datasets/retrieval_qa.json                      # labeled QA set
data/knowledge/general_ai_trends.md, pricing_and_gtm.md  # distractors + depth
src/ai_research_agent/evaluation/retrieval_eval.py    # IR metrics + benchmark
src/ai_research_agent/evaluation/ragas_eval.py        # Ragas answer eval
src/ai_research_agent/evaluation/_compat.py           # ragas/langchain-v1 shim
src/ai_research_agent/cli.py                           # + benchmark, evaluate
examples/11_evaluation/retrieval_benchmark.py
tests/evaluation/test_retrieval_eval.py
pyproject.toml                                         # + ragas, datasets
```

## Commands to run (on camera)

```bash
uv run pytest -q
uv run python examples/11_evaluation/retrieval_benchmark.py
uv run research-agent benchmark --k 4
# With an LLM key in .env:
uv run research-agent evaluate --k 4       # Ragas faithfulness/relevancy/...
```

## Demo moment

Run `benchmark`, read the table, and make the honest call on camera: "hybrid is
our robust default; we're not shipping the reranker for *this* corpus because
the numbers don't justify it." That's the loop working. Then, with a key, run
`evaluate` to show Ragas scoring the *answers*, not just the retrieval.

## Next phase

`phase-11` — LangChain tools (`@tool`, schemas, tool calling).
