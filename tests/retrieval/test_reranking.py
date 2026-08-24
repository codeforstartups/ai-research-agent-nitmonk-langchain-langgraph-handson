"""Phase 09 — reranking (deterministic lexical fallback, no network)."""

from __future__ import annotations

from langchain_core.documents import Document

from ai_research_agent.reranking import Reranker, RerankingRetriever
from ai_research_agent.retrieval.bm25 import BM25Retriever

DOCS = [
    Document(page_content="unrelated content about plants and soil", metadata={"source": "a", "chunk": 0}),
    Document(page_content="enterprise buyers require SOC2 audit logs and IP indemnity", metadata={"source": "b", "chunk": 0}),
    Document(page_content="a short note about weather", metadata={"source": "c", "chunk": 0}),
]


def test_reranker_promotes_most_relevant() -> None:
    reranker = Reranker(use_flashrank=False)  # deterministic lexical path
    ranked = reranker.rerank("SOC2 audit logs IP indemnity", DOCS, top_n=3)
    assert ranked[0].metadata["source"] == "b"


def test_reranker_top_n_limits_results() -> None:
    reranker = Reranker(use_flashrank=False)
    assert len(reranker.rerank("enterprise", DOCS, top_n=1)) == 1


def test_reranker_handles_empty() -> None:
    assert Reranker(use_flashrank=False).rerank("q", [], top_n=3) == []


def test_reranking_retriever_narrows_candidates() -> None:
    base = BM25Retriever.from_documents(DOCS, k=3)
    retriever = RerankingRetriever(
        base_retriever=base, reranker=Reranker(use_flashrank=False), k=1
    )
    results = retriever.invoke("SOC2 audit logs")
    assert len(results) == 1
    assert results[0].metadata["source"] == "b"
