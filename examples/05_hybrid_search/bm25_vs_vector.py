"""Phase 07 — BM25 keyword retrieval vs dense vector retrieval.

Run:
    uv run python examples/05_hybrid_search/bm25_vs_vector.py

Teaches:
* BM25 shines on exact terms/rare tokens; vectors shine on paraphrase,
* why neither alone is enough — motivating hybrid fusion in phase 08.
"""

from __future__ import annotations

from ai_research_agent.config import get_settings
from ai_research_agent.ingestion import chunk_documents, load_directory
from ai_research_agent.retrieval import BM25Retriever, build_vectorstore, get_retriever
from ai_research_agent.utils import console


def show(title: str, docs) -> None:
    console.rule(f"[bold]{title}[/]")
    for i, doc in enumerate(docs, 1):
        cite = f"{doc.metadata['source']}#{doc.metadata['chunk']}"
        console.print(f"{i}. [cyan]{cite}[/] — {doc.page_content[:90].strip()}…")


def main() -> None:
    chunks = chunk_documents(load_directory(get_settings().knowledge_dir))

    bm25 = BM25Retriever.from_documents(chunks, k=3)
    vector = get_retriever(build_vectorstore(chunks), k=3)

    # A query with an exact rare term ("SOC2") plus paraphraseable intent.
    query = "SOC2 compliance requirements"
    console.print(f"Query: [italic]{query}[/]\n")
    show("BM25 (keyword)", bm25.invoke(query))
    show("Vector (semantic)", vector.invoke(query))


if __name__ == "__main__":
    main()
