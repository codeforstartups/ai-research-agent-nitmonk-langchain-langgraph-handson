"""Vector store: persist chunk embeddings and retrieve by similarity.

**Product store: Pinecone (serverless).** Pinecone hosts and indexes the
embeddings in the cloud, so ingestion happens once and any process can query the
same index. It has no local mode, so the product path requires
``PINECONE_API_KEY``.

**Teaching double: in-memory.** The examples and tests use langchain-core's
``InMemoryVectorStore`` via :func:`build_ephemeral_retriever` — a throwaway,
no-key store so the concepts stay runnable offline. Same ``Retriever`` interface
either way, so every downstream component (RAG, hybrid, agent) is unaffected.

The embedding stays pluggable (fastembed/OpenAI/Google); the Pinecone index is
created sized to whatever embedding you configure.
"""

from __future__ import annotations

import time

from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.vectorstores import (
    InMemoryVectorStore,
    VectorStore,
    VectorStoreRetriever,
)

from ai_research_agent.config import get_settings
from ai_research_agent.embeddings import get_embeddings
from ai_research_agent.ingestion import chunk_documents, load_directory


class MissingPineconeError(RuntimeError):
    """Raised when the Pinecone product path is used without a key."""


def _embedding_dimension(embeddings: Embeddings) -> int:
    return len(embeddings.embed_query("dimension probe"))


def _pinecone_client():
    settings = get_settings()
    if not settings.has_pinecone:
        raise MissingPineconeError(
            "PINECONE_API_KEY is not set. The product retrieval path uses "
            "Pinecone; set the key in .env, or use build_ephemeral_retriever() "
            "for the offline in-memory store."
        )
    from pinecone import Pinecone

    return Pinecone(api_key=settings.pinecone_api_key)


def ensure_index(dimension: int, *, index_name: str | None = None):
    """Create the Pinecone serverless index if it doesn't exist; return it."""
    from pinecone import ServerlessSpec

    settings = get_settings()
    index_name = index_name or settings.pinecone_index
    pc = _pinecone_client()

    existing = {idx["name"] for idx in pc.list_indexes()}
    if index_name not in existing:
        pc.create_index(
            name=index_name,
            dimension=dimension,
            metric="cosine",
            spec=ServerlessSpec(cloud=settings.pinecone_cloud, region=settings.pinecone_region),
        )
        # Wait for the index to become ready before we write to it.
        while not pc.describe_index(index_name).status["ready"]:
            time.sleep(1)
    return pc.Index(index_name)


def build_vectorstore(
    chunks: list[Document],
    *,
    embeddings: Embeddings | None = None,
    index_name: str | None = None,
    namespace: str | None = None,
) -> VectorStore:
    """Create/populate the Pinecone index from ``chunks`` and return the store."""
    from langchain_pinecone import PineconeVectorStore

    settings = get_settings()
    embeddings = embeddings or get_embeddings()
    index_name = index_name or settings.pinecone_index

    index = ensure_index(_embedding_dimension(embeddings), index_name=index_name)
    store = PineconeVectorStore(index=index, embedding=embeddings, namespace=namespace)
    if chunks:
        store.add_documents(chunks)
    return store


def load_vectorstore(
    *,
    embeddings: Embeddings | None = None,
    index_name: str | None = None,
    namespace: str | None = None,
) -> VectorStore:
    """Open the existing persisted Pinecone index."""
    from langchain_pinecone import PineconeVectorStore

    settings = get_settings()
    embeddings = embeddings or get_embeddings()
    index_name = index_name or settings.pinecone_index
    # Touch the client to fail fast with a clear error if the key is missing.
    _pinecone_client()
    return PineconeVectorStore.from_existing_index(
        index_name=index_name, embedding=embeddings, namespace=namespace
    )


def build_ephemeral_retriever(
    chunks: list[Document],
    *,
    embeddings: Embeddings | None = None,
    k: int = 4,
) -> VectorStoreRetriever:
    """Build a no-key, in-memory retriever (teaching/testing double)."""
    embeddings = embeddings or get_embeddings()
    store = InMemoryVectorStore(embedding=embeddings)
    store.add_documents(chunks)
    return store.as_retriever(search_kwargs={"k": k})


def get_retriever(store: VectorStore, *, k: int = 4) -> VectorStoreRetriever:
    """Return a LangChain retriever over ``store`` returning top-``k`` chunks."""
    return store.as_retriever(search_kwargs={"k": k})


def index_knowledge_base(
    *,
    embeddings: Embeddings | None = None,
    index_name: str | None = None,
) -> tuple[int, int]:
    """Load → chunk → index the configured knowledge base into Pinecone.

    Returns ``(num_documents, num_chunks)``.
    """
    settings = get_settings()
    documents = load_directory(settings.knowledge_dir)
    chunks = chunk_documents(documents)
    build_vectorstore(chunks, embeddings=embeddings, index_name=index_name)
    return len(documents), len(chunks)
