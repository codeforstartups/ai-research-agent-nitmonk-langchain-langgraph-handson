"""Phase 10 — retrieval benchmark metrics + Ragas guard (offline)."""

from __future__ import annotations

from langchain_core.documents import Document

from ai_research_agent.evaluation import RagasUnavailableError, run_benchmark
from ai_research_agent.evaluation.retrieval_eval import score_query


def _doc(source: str) -> Document:
    return Document(page_content=source, metadata={"source": source, "chunk": 0})


def test_score_query_perfect_hit() -> None:
    hit, recall, rr = score_query([_doc("a.md"), _doc("b.md")], {"a.md"})
    assert hit == 1.0
    assert recall == 1.0
    assert rr == 1.0  # relevant doc is rank 1


def test_score_query_relevant_at_rank_two() -> None:
    hit, recall, rr = score_query([_doc("x.md"), _doc("a.md")], {"a.md"})
    assert hit == 1.0
    assert rr == 0.5  # 1 / 2


def test_score_query_miss() -> None:
    hit, recall, rr = score_query([_doc("x.md")], {"a.md"})
    assert (hit, recall, rr) == (0.0, 0.0, 0.0)


def test_benchmark_runs_all_configs_offline() -> None:
    from ai_research_agent.evaluation.retrieval_eval import load_dataset

    n_items = len(load_dataset())
    results = run_benchmark(k=4)
    assert set(results) == {"vector", "bm25", "hybrid", "hybrid+rerank"}
    # Every config should retrieve *something* relevant on this easy set.
    assert all(m.hit_rate > 0 for m in results.values())
    assert all(m.n == n_items for m in results.values())


def test_ragas_requires_llm_key() -> None:
    from ai_research_agent.config import get_settings

    s = get_settings()
    if s.has_anthropic or s.has_openai or s.has_google:
        return  # a key is present in this environment; the guard doesn't apply

    from ai_research_agent.evaluation import run_ragas_eval

    try:
        run_ragas_eval()
    except RagasUnavailableError as exc:
        assert "LLM judge" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("expected RagasUnavailableError")
