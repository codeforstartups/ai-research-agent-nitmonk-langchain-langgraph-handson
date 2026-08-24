"""Phase 08 — reciprocal-rank fusion and hybrid retrieval."""

from __future__ import annotations

from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever

from ai_research_agent.retrieval import reciprocal_rank_fusion
from ai_research_agent.retrieval.hybrid import HybridRetriever


def _d(source: str) -> Document:
    return Document(page_content=source, metadata={"source": source, "chunk": 0})


A, B, C, D = _d("a"), _d("b"), _d("c"), _d("d")


def test_rrf_boosts_docs_ranked_by_both_lists() -> None:
    # B is top of list 2 and 2nd of list 1 -> should win overall.
    fused = reciprocal_rank_fusion([[A, B, C], [B, D]], top_k=4)
    assert fused[0].metadata["source"] == "b"


def test_rrf_unions_and_dedupes() -> None:
    fused = reciprocal_rank_fusion([[A, B], [B, C]], top_k=10)
    sources = [d.metadata["source"] for d in fused]
    assert sorted(sources) == ["a", "b", "c"]  # no duplicate 'b'


def test_rrf_respects_top_k() -> None:
    fused = reciprocal_rank_fusion([[A, B, C, D]], top_k=2)
    assert len(fused) == 2


class _FixedRetriever(BaseRetriever):
    """Returns a fixed doc list — for testing HybridRetriever wiring."""

    docs: list[Document]

    def _get_relevant_documents(self, query, *, run_manager):  # noqa: ARG002
        return self.docs


def test_hybrid_retriever_fuses_sub_retrievers() -> None:
    hybrid = HybridRetriever(
        retrievers=[_FixedRetriever(docs=[A, B]), _FixedRetriever(docs=[B, C])], k=3
    )
    results = hybrid.invoke("q")
    assert results[0].metadata["source"] == "b"  # in both
    assert {d.metadata["source"] for d in results} == {"a", "b", "c"}
