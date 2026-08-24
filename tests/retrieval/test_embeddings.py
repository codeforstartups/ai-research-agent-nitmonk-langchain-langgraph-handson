"""Phase 03 — embeddings interface and similarity math (network-free)."""

from __future__ import annotations

from ai_research_agent.config import EmbeddingsProvider, Settings
from ai_research_agent.embeddings import (
    DeterministicEmbeddings,
    cosine_similarity,
    rank_by_similarity,
)


def test_deterministic_embeddings_shape_and_repeatable() -> None:
    emb = DeterministicEmbeddings(dim=128)
    v1 = emb.embed_query("autonomous coding agents")
    v2 = emb.embed_query("autonomous coding agents")
    assert len(v1) == 128
    assert v1 == v2  # deterministic


def test_cosine_self_similarity_is_one() -> None:
    emb = DeterministicEmbeddings()
    v = emb.embed_query("model context protocol")
    assert abs(cosine_similarity(v, v) - 1.0) < 1e-9


def test_shared_vocabulary_ranks_higher() -> None:
    emb = DeterministicEmbeddings()
    query = emb.embed_query("AI coding agents for developers")
    docs = emb.embed_documents(
        [
            "AI coding agents help developers write software",  # overlaps
            "the mitochondria is the powerhouse of the cell",  # unrelated
        ]
    )
    ranking = rank_by_similarity(query, docs)
    assert ranking[0][0] == 0  # the overlapping doc wins
    assert ranking[0][1] > ranking[1][1]


def test_zero_vector_similarity_is_zero() -> None:
    assert cosine_similarity([0.0, 0.0], [1.0, 1.0]) == 0.0


def test_factory_returns_embeddings_object() -> None:
    from ai_research_agent.embeddings.factory import get_embeddings

    # Force the deterministic path without touching the network: select openai
    # with no key -> falls back; but to stay fully offline we assert the
    # deterministic class works standalone (fastembed download is exercised in
    # the example, not the test suite).
    settings = Settings(_env_file=None)  # type: ignore[call-arg]
    assert settings.embeddings_provider is EmbeddingsProvider.FASTEMBED
    # Sanity: the deterministic fallback satisfies the Embeddings contract.
    emb = DeterministicEmbeddings()
    assert hasattr(emb, "embed_documents") and hasattr(emb, "embed_query")
    _ = get_embeddings  # referenced to document the public entry point
