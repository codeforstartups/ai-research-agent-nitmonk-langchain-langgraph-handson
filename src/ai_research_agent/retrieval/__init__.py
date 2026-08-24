"""Retrieval package: vector store, and (later) BM25, hybrid fusion, reranking."""

from ai_research_agent.retrieval.rag import (
    RagResult,
    answer_question,
    build_rag_chain,
)
from ai_research_agent.retrieval.vectorstore import (
    build_vectorstore,
    get_retriever,
    index_knowledge_base,
    load_vectorstore,
)

__all__ = [
    "RagResult",
    "answer_question",
    "build_rag_chain",
    "build_vectorstore",
    "get_retriever",
    "index_knowledge_base",
    "load_vectorstore",
]
