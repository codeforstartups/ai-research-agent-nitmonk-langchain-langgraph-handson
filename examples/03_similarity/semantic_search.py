"""Phase 03 — semantic similarity search (no vector DB yet).

Run:
    uv run python examples/03_similarity/semantic_search.py

Teaches:
* cosine similarity as the ranking signal,
* "semantic" search: a query matches by *meaning*, not exact keywords,
* the naive O(n) approach we'll replace with a vector store in phase-05.
"""

from __future__ import annotations

from ai_research_agent.embeddings import get_embeddings, rank_by_similarity
from ai_research_agent.utils import console

DOCUMENTS = [
    "Cursor and GitHub Copilot are AI pair-programming tools.",
    "The capital of France is Paris.",
    "Enterprises worry about code security and IP when adopting AI agents.",
    "Photosynthesis converts sunlight into chemical energy in plants.",
    "Devin markets itself as an autonomous AI software engineer.",
]

QUERY = "Which companies build autonomous coding agents for developers?"


def main() -> None:
    embeddings = get_embeddings()
    console.print(f"[dim]backend:[/] {type(embeddings).__name__}\n")

    doc_vecs = embeddings.embed_documents(DOCUMENTS)
    query_vec = embeddings.embed_query(QUERY)

    console.rule("[bold]Ranked by cosine similarity[/]")
    console.print(f"Query: [italic]{QUERY}[/]\n")
    for rank, (idx, score) in enumerate(rank_by_similarity(query_vec, doc_vecs), 1):
        console.print(f"{rank}. ({score:+.3f})  {DOCUMENTS[idx]}")


if __name__ == "__main__":
    main()
