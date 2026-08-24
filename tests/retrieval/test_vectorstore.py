"""Phase 05 — vector store build + retrieve (in-memory, deterministic)."""

from __future__ import annotations

from langchain_core.documents import Document

from ai_research_agent.embeddings import DeterministicEmbeddings
from ai_research_agent.retrieval import build_vectorstore, get_retriever

DOCS = [
    Document(
        page_content="Enterprise buyers require SOC2, IP indemnity, and audit logs.",
        metadata={"source": "enterprise.md", "chunk": 0},
    ),
    Document(
        page_content="Photosynthesis converts sunlight into chemical energy.",
        metadata={"source": "biology.md", "chunk": 0},
    ),
    Document(
        page_content="Cursor and Copilot are AI pair-programming tools.",
        metadata={"source": "tools.md", "chunk": 0},
    ),
]


def _store():
    return build_vectorstore(DOCS, embeddings=DeterministicEmbeddings(), persist_directory=None)


def test_retrieve_returns_relevant_chunk_first() -> None:
    retriever = get_retriever(_store(), k=2)
    # DeterministicEmbeddings are lexical, so the query overlaps the target's
    # vocabulary (SOC2 / audit / logs / enterprise buyers require).
    results = retriever.invoke("SOC2 audit logs that enterprise buyers require")
    assert results
    assert results[0].metadata["source"] == "enterprise.md"


def test_retriever_respects_k() -> None:
    retriever = get_retriever(_store(), k=1)
    assert len(retriever.invoke("coding tools")) == 1


def test_metadata_survives_round_trip() -> None:
    results = get_retriever(_store(), k=3).invoke("anything")
    assert all("source" in d.metadata and "chunk" in d.metadata for d in results)
