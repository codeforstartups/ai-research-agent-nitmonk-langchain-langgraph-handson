"""Embeddings package: pluggable embeddings + similarity math."""

from ai_research_agent.embeddings.deterministic import DeterministicEmbeddings
from ai_research_agent.embeddings.factory import get_embeddings
from ai_research_agent.embeddings.fastembed_local import FastEmbedEmbeddings
from ai_research_agent.embeddings.similarity import cosine_similarity, rank_by_similarity

__all__ = [
    "DeterministicEmbeddings",
    "FastEmbedEmbeddings",
    "cosine_similarity",
    "get_embeddings",
    "rank_by_similarity",
]
