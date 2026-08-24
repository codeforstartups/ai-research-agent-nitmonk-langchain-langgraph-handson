"""Phase 06 — RAG chain plumbing (offline model, deterministic embeddings)."""

from __future__ import annotations

from langchain_core.documents import Document

from ai_research_agent.embeddings import DeterministicEmbeddings
from ai_research_agent.models import OfflineChatModel
from ai_research_agent.prompts import format_docs
from ai_research_agent.retrieval import build_ephemeral_retriever, build_rag_chain

DOCS = [
    Document(
        page_content="Enterprise buyers require SOC2, IP indemnity, and audit logs.",
        metadata={"source": "enterprise.md", "chunk": 0},
    ),
    Document(
        page_content="Reliability at task completion is the dominant differentiator.",
        metadata={"source": "market.md", "chunk": 0},
    ),
]


def test_format_docs_includes_citations() -> None:
    rendered = format_docs(DOCS)
    assert "[enterprise.md#0]" in rendered
    assert "[market.md#0]" in rendered


def test_rag_chain_returns_answer_and_sources() -> None:
    retriever = build_ephemeral_retriever(DOCS, embeddings=DeterministicEmbeddings(), k=2)
    chain = build_rag_chain(retriever, model=OfflineChatModel())
    result = chain.invoke("What do enterprise buyers require?")

    assert set(result.keys()) == {"answer", "sources"}
    assert isinstance(result["answer"], str) and result["answer"]
    assert len(result["sources"]) == 2
    assert all(isinstance(d, Document) for d in result["sources"])
