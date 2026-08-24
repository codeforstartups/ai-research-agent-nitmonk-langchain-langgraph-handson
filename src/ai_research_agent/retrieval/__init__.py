"""Retrieval package: vector store, and (later) BM25, hybrid fusion, reranking."""

from ai_research_agent.retrieval.bm25 import BM25Retriever, tokenize
from ai_research_agent.retrieval.hybrid import (
    HybridRetriever,
    build_hybrid_from_documents,
    reciprocal_rank_fusion,
)
from ai_research_agent.retrieval.rag import (
    RagResult,
    answer_question,
    build_rag_chain,
)
from ai_research_agent.retrieval.vectorstore import (
    build_ephemeral_retriever,
    build_vectorstore,
    get_retriever,
    index_knowledge_base,
    load_vectorstore,
)

__all__ = [
    "BM25Retriever",
    "HybridRetriever",
    "RagResult",
    "answer_question",
    "build_ephemeral_retriever",
    "build_hybrid_from_documents",
    "build_rag_chain",
    "build_vectorstore",
    "get_retriever",
    "index_knowledge_base",
    "load_vectorstore",
    "reciprocal_rank_fusion",
    "tokenize",
]
