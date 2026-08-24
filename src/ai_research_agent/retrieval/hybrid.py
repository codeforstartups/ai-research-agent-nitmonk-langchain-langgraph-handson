"""Hybrid retrieval via Reciprocal Rank Fusion (RRF).

Phase 07 showed BM25 and vector search each find things the other misses. RRF is
a simple, robust way to merge ranked lists *without* needing their scores to be
on comparable scales: a document's fused score is the sum, over every retriever
that returned it, of ``1 / (rrf_k + rank)``.

Consequences that make RRF a great default:
* documents ranked highly by *either* retriever surface,
* documents ranked highly by *both* get a strong boost,
* only ranks matter, so incompatible score scales are a non-issue.
"""

from __future__ import annotations

from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from pydantic import ConfigDict


def _doc_key(doc: Document) -> str:
    """Stable identity for de-duplication across retrievers."""
    source = doc.metadata.get("source")
    chunk = doc.metadata.get("chunk")
    if source is not None and chunk is not None:
        return f"{source}#{chunk}"
    return doc.page_content


def reciprocal_rank_fusion(
    ranked_lists: list[list[Document]],
    *,
    rrf_k: int = 60,
    top_k: int = 4,
) -> list[Document]:
    """Fuse multiple ranked document lists into one, by RRF."""
    scores: dict[str, float] = {}
    docs: dict[str, Document] = {}
    for ranked in ranked_lists:
        for rank, doc in enumerate(ranked):
            key = _doc_key(doc)
            scores[key] = scores.get(key, 0.0) + 1.0 / (rrf_k + rank)
            docs.setdefault(key, doc)
    ordered = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
    return [docs[key] for key, _ in ordered[:top_k]]


class HybridRetriever(BaseRetriever):
    """Combine several retrievers into one via reciprocal-rank fusion."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    retrievers: list[BaseRetriever]
    k: int = 4
    rrf_k: int = 60

    def _get_relevant_documents(
        self, query: str, *, run_manager: CallbackManagerForRetrieverRun
    ) -> list[Document]:
        ranked_lists = [r.invoke(query) for r in self.retrievers]
        return reciprocal_rank_fusion(ranked_lists, rrf_k=self.rrf_k, top_k=self.k)


def build_hybrid_from_documents(
    chunks: list[Document], *, k: int = 4, per_retriever_k: int = 8
) -> HybridRetriever:
    """Build a vector+BM25 hybrid retriever from a list of chunks.

    Each underlying retriever fetches ``per_retriever_k`` candidates (a wider net)
    which RRF then fuses down to the final ``k``.
    """
    from ai_research_agent.retrieval.bm25 import BM25Retriever
    from ai_research_agent.retrieval.vectorstore import build_ephemeral_retriever

    # Builds throwaway in-memory indexes over the given chunks (no Pinecone key
    # needed). The production hybrid path pairs a Pinecone vector retriever with
    # a BM25 built from the same corpus.
    vector = build_ephemeral_retriever(chunks, k=per_retriever_k)
    bm25 = BM25Retriever.from_documents(chunks, k=per_retriever_k)
    return HybridRetriever(retrievers=[vector, bm25], k=k)
