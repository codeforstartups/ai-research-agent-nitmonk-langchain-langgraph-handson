"""Phase 05 — index chunks into a vector store and retrieve.

Run:
    uv run python examples/04_rag/vector_search.py

Teaches:
* persisting embeddings in a vector DB (Pinecone in production) instead of an
  in-memory loop — this demo uses the offline in-memory double,
* the LangChain ``Retriever`` abstraction (``.invoke(query)`` → documents),
* traceable results (each hit still carries ``source`` + ``chunk``).
"""

from __future__ import annotations

from ai_research_agent.config import get_settings
from ai_research_agent.ingestion import chunk_documents, load_directory
from ai_research_agent.retrieval import build_ephemeral_retriever
from ai_research_agent.utils import console

QUERY = "What do enterprise buyers require before adopting an AI coding agent?"


def main() -> None:
    settings = get_settings()

    # Build a no-key in-memory index for the demo. The product CLI
    # (research-agent ingest/search) uses the persistent Pinecone index instead.
    documents = load_directory(settings.knowledge_dir)
    chunks = chunk_documents(documents)
    retriever = build_ephemeral_retriever(chunks, k=3)
    console.print(f"Indexed {len(chunks)} chunks into an in-memory store.\n")
    console.rule("[bold]Top 3 retrieved chunks[/]")
    console.print(f"Query: [italic]{QUERY}[/]\n")
    for i, doc in enumerate(retriever.invoke(QUERY), 1):
        cite = f"{doc.metadata['source']}#{doc.metadata['chunk']}"
        console.print(f"[bold]{i}. [cyan]{cite}[/][/]")
        console.print(f"   {doc.page_content[:200].strip()}…\n")


if __name__ == "__main__":
    main()
