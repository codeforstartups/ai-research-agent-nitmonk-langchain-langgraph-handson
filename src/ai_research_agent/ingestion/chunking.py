"""Split documents into retrieval-sized chunks.

Why chunk at all? Embedding a whole document blurs its meaning into one vector,
and stuffing huge documents into a prompt is wasteful and often exceeds context
limits. We split into overlapping windows so each chunk is a coherent, embeddable
unit, and the overlap keeps ideas that straddle a boundary retrievable.

We use ``RecursiveCharacterTextSplitter`` because it tries to break on natural
boundaries (paragraphs → lines → words) before resorting to hard cuts.
"""

from __future__ import annotations

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

DEFAULT_CHUNK_SIZE = 800
DEFAULT_CHUNK_OVERLAP = 120


def chunk_documents(
    documents: list[Document],
    *,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
) -> list[Document]:
    """Split ``documents`` into overlapping chunks, preserving+enriching metadata.

    Each resulting chunk keeps its parent's metadata and gains a ``chunk`` index
    so it can be cited and de-duplicated downstream.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    chunks = splitter.split_documents(documents)

    # Annotate per-source chunk index for traceable citations.
    per_source_counter: dict[str, int] = {}
    for chunk in chunks:
        source = chunk.metadata.get("source", "unknown")
        idx = per_source_counter.get(source, 0)
        chunk.metadata["chunk"] = idx
        per_source_counter[source] = idx + 1
    return chunks
