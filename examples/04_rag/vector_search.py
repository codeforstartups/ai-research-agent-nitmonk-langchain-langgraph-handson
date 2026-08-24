"""Phase 05 — index chunks into a vector store and retrieve.

Run:
    uv run python examples/04_rag/vector_search.py

Teaches:
* persisting embeddings in a vector DB (Chroma) instead of an in-memory loop,
* the LangChain ``Retriever`` abstraction (``.invoke(query)`` → documents),
* traceable results (each hit still carries ``source`` + ``chunk``).
"""

from __future__ import annotations

from ai_research_agent.config import get_settings
from ai_research_agent.ingestion import chunk_documents, load_directory
from ai_research_agent.retrieval import build_vectorstore, get_retriever
from ai_research_agent.utils import console

QUERY = "What do enterprise buyers require before adopting an AI coding agent?"


def main() -> None:
    settings = get_settings()

    # Build an in-memory index for the demo (no disk writes).
    documents = load_directory(settings.knowledge_dir)
    chunks = chunk_documents(documents)
    store = build_vectorstore(chunks)  # persist_directory=None -> ephemeral
    console.print(f"Indexed {len(chunks)} chunks into an in-memory Chroma store.\n")

    retriever = get_retriever(store, k=3)
    console.rule("[bold]Top 3 retrieved chunks[/]")
    console.print(f"Query: [italic]{QUERY}[/]\n")
    for i, doc in enumerate(retriever.invoke(QUERY), 1):
        cite = f"{doc.metadata['source']}#{doc.metadata['chunk']}"
        console.print(f"[bold]{i}. [cyan]{cite}[/][/]")
        console.print(f"   {doc.page_content[:200].strip()}…\n")


if __name__ == "__main__":
    main()
