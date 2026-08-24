"""Vector store: persist chunk embeddings and retrieve by similarity.

Phase 03 ranked vectors with a hand-written O(n) loop. That does not scale and
forgets everything between runs. A **vector database** (Chroma here) stores the
embeddings, indexes them for fast approximate nearest-neighbour search, and
persists to disk so ingestion happens once.

We keep the *embedding* pluggable: Chroma is given our `get_embeddings()` model,
so the same store works with fastembed, OpenAI or Google vectors.
"""

from __future__ import annotations

from pathlib import Path

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.vectorstores import VectorStoreRetriever

from ai_research_agent.config import get_settings
from ai_research_agent.embeddings import get_embeddings
from ai_research_agent.ingestion import chunk_documents, load_directory

DEFAULT_COLLECTION = "knowledge"


def build_vectorstore(
    chunks: list[Document],
    *,
    embeddings: Embeddings | None = None,
    persist_directory: Path | str | None = None,
    collection_name: str = DEFAULT_COLLECTION,
) -> Chroma:
    """Create (or overwrite) a Chroma collection from ``chunks``.

    ``persist_directory=None`` yields an ephemeral in-memory store (used in
    tests); a path persists the index to disk.
    """
    embeddings = embeddings or get_embeddings()
    store = Chroma(
        collection_name=collection_name,
        embedding_function=embeddings,
        persist_directory=str(persist_directory) if persist_directory else None,
    )
    # Start clean so re-ingesting is idempotent, not additive.
    try:
        store.reset_collection()
    except Exception:  # pragma: no cover - older chroma without reset
        pass
    if chunks:
        store.add_documents(chunks)
    return store


def load_vectorstore(
    *,
    embeddings: Embeddings | None = None,
    persist_directory: Path | str | None = None,
    collection_name: str = DEFAULT_COLLECTION,
) -> Chroma:
    """Open an existing persisted Chroma collection."""
    settings = get_settings()
    embeddings = embeddings or get_embeddings()
    persist_directory = persist_directory or settings.vectorstore_dir
    return Chroma(
        collection_name=collection_name,
        embedding_function=embeddings,
        persist_directory=str(persist_directory),
    )


def get_retriever(store: Chroma, *, k: int = 4) -> VectorStoreRetriever:
    """Return a LangChain retriever over ``store`` returning top-``k`` chunks."""
    return store.as_retriever(search_kwargs={"k": k})


def index_knowledge_base(
    *,
    persist_directory: Path | str | None = None,
    embeddings: Embeddings | None = None,
) -> tuple[int, int]:
    """Load → chunk → index the configured knowledge base to disk.

    Returns ``(num_documents, num_chunks)``.
    """
    settings = get_settings()
    persist_directory = persist_directory or settings.vectorstore_dir
    Path(persist_directory).mkdir(parents=True, exist_ok=True)

    documents = load_directory(settings.knowledge_dir)
    chunks = chunk_documents(documents)
    build_vectorstore(chunks, embeddings=embeddings, persist_directory=persist_directory)
    return len(documents), len(chunks)
