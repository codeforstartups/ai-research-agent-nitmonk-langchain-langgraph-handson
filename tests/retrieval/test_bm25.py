"""Phase 07 — BM25 keyword retrieval."""

from __future__ import annotations

from langchain_core.documents import Document

from ai_research_agent.retrieval import BM25Retriever, tokenize

DOCS = [
    Document(page_content="SOC2 and audit logs are enterprise requirements.", metadata={"source": "a", "chunk": 0}),
    Document(page_content="Reliability is the dominant differentiator.", metadata={"source": "b", "chunk": 0}),
    Document(page_content="Photosynthesis happens in plants.", metadata={"source": "c", "chunk": 0}),
]


def test_tokenize_lowercases_and_splits() -> None:
    assert tokenize("SOC2, audit-logs!") == ["soc2", "audit", "logs"]


def test_bm25_finds_exact_keyword() -> None:
    retriever = BM25Retriever.from_documents(DOCS, k=2)
    results = retriever.invoke("SOC2 audit logs")
    assert results[0].metadata["source"] == "a"


def test_bm25_returns_empty_for_no_overlap() -> None:
    retriever = BM25Retriever.from_documents(DOCS, k=3)
    # Query with zero shared vocabulary -> all scores 0 -> filtered out.
    assert retriever.invoke("quantum chromodynamics") == []


def test_bm25_respects_k() -> None:
    retriever = BM25Retriever.from_documents(DOCS, k=1)
    assert len(retriever.invoke("enterprise reliability plants")) <= 1
