"""Phase 08 — hybrid retrieval via reciprocal-rank fusion.

Run:
    uv run python examples/05_hybrid_search/hybrid_fusion.py

Teaches:
* fusing BM25 + vector rankings with RRF (ranks, not scores),
* why hybrid is the robust default: it recovers both exact-term and semantic
  hits.
"""

from __future__ import annotations

from ai_research_agent.config import get_settings
from ai_research_agent.ingestion import chunk_documents, load_directory
from ai_research_agent.retrieval import (
    BM25Retriever,
    build_ephemeral_retriever,
    build_hybrid_from_documents,
)
from ai_research_agent.utils import console


def show(title: str, docs) -> None:
    console.rule(f"[bold]{title}[/]")
    for i, doc in enumerate(docs, 1):
        cite = f"{doc.metadata['source']}#{doc.metadata['chunk']}"
        console.print(f"{i}. [cyan]{cite}[/]")


def main() -> None:
    chunks = chunk_documents(load_directory(get_settings().knowledge_dir))
    query = "What are the SOC2 and IP indemnity moats for a new entrant?"
    console.print(f"Query: [italic]{query}[/]\n")

    show("BM25 only", BM25Retriever.from_documents(chunks, k=4).invoke(query))
    show("Vector only", build_ephemeral_retriever(chunks, k=4).invoke(query))
    show("Hybrid (RRF)", build_hybrid_from_documents(chunks, k=4).invoke(query))


if __name__ == "__main__":
    main()
