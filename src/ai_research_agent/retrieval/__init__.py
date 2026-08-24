"""Retrieval package: vector store, and (later) BM25, hybrid fusion, reranking."""

from ai_research_agent.retrieval.vectorstore import (
    build_vectorstore,
    get_retriever,
    index_knowledge_base,
    load_vectorstore,
)

__all__ = [
    "build_vectorstore",
    "get_retriever",
    "index_knowledge_base",
    "load_vectorstore",
]
