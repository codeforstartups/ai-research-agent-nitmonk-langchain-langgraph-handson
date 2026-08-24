"""Phase 09 — retrieve-then-rerank.

Run:
    uv run python examples/05_hybrid_search/reranking.py

Teaches:
* the two-stage pattern: cheap wide retrieval → expensive precise reranking,
* how a cross-encoder reorders candidates by true query-document relevance.

First run downloads a small FlashRank model; offline it falls back to a lexical
reranker (announced, not silent).
"""

from __future__ import annotations

from ai_research_agent.config import get_settings
from ai_research_agent.ingestion import chunk_documents, load_directory
from ai_research_agent.reranking import Reranker, RerankingRetriever
from ai_research_agent.retrieval import build_hybrid_from_documents
from ai_research_agent.utils import console


def show(title: str, docs) -> None:
    console.rule(f"[bold]{title}[/]")
    for i, doc in enumerate(docs, 1):
        cite = f"{doc.metadata['source']}#{doc.metadata['chunk']}"
        console.print(f"{i}. [cyan]{cite}[/] — {doc.page_content[:80].strip()}…")


def main() -> None:
    chunks = chunk_documents(load_directory(get_settings().knowledge_dir))
    query = "What governance controls do regulated enterprises demand?"
    console.print(f"Query: [italic]{query}[/]\n")

    hybrid = build_hybrid_from_documents(chunks, k=6)
    show("Hybrid candidates (k=6)", hybrid.invoke(query))

    reranked = RerankingRetriever(base_retriever=hybrid, reranker=Reranker(), k=3)
    show("After reranking (top 3)", reranked.invoke(query))


if __name__ == "__main__":
    main()
