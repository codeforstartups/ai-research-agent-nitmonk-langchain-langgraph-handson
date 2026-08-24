"""A retriever that reranks another retriever's output.

Pattern: cast a wide net cheaply (base retriever fetches many candidates), then
spend the expensive cross-encoder only on that shortlist to produce a precise
top-``k``. This two-stage "retrieve-then-rerank" is the standard high-quality
RAG retrieval setup.
"""

from __future__ import annotations

from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from pydantic import ConfigDict

from ai_research_agent.reranking.reranker import Reranker


class RerankingRetriever(BaseRetriever):
    """Wrap a base retriever with a cross-encoder reranking stage."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    base_retriever: BaseRetriever
    reranker: Reranker
    k: int = 4

    def _get_relevant_documents(
        self, query: str, *, run_manager: CallbackManagerForRetrieverRun
    ) -> list[Document]:
        candidates = self.base_retriever.invoke(query)
        return self.reranker.rerank(query, candidates, top_n=self.k)
