"""Phase 05 — vector store build + retrieve (in-memory double, deterministic).

The product store is Pinecone (needs a key/network); these tests exercise the
same ``Retriever`` interface through the offline in-memory double.
"""

from __future__ import annotations

from langchain_core.documents import Document

from ai_research_agent.embeddings import DeterministicEmbeddings
from ai_research_agent.retrieval import build_ephemeral_retriever

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


def _retriever(k: int):
    return build_ephemeral_retriever(DOCS, embeddings=DeterministicEmbeddings(), k=k)


def test_retrieve_returns_relevant_chunk_first() -> None:
    retriever = _retriever(2)
    # DeterministicEmbeddings are lexical, so the query overlaps the target's
    # vocabulary (SOC2 / audit / logs / enterprise buyers require).
    results = retriever.invoke("SOC2 audit logs that enterprise buyers require")
    assert results
    assert results[0].metadata["source"] == "enterprise.md"


def test_retriever_respects_k() -> None:
    assert len(_retriever(1).invoke("coding tools")) == 1


def test_metadata_survives_round_trip() -> None:
    results = _retriever(3).invoke("anything")
    assert all("source" in d.metadata and "chunk" in d.metadata for d in results)
