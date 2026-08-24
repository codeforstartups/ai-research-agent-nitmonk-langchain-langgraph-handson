"""Phase 03 — turning text into vectors.

Run:
    uv run python examples/02_embeddings/basic_embeddings.py

Teaches:
* the ``Embeddings`` interface (``embed_documents`` / ``embed_query``),
* what an embedding actually is (a list of floats — a point in meaning-space),
* that the same interface backs local (fastembed) and hosted (OpenAI/Google)
  models.
"""

from __future__ import annotations

from ai_research_agent.embeddings import get_embeddings
from ai_research_agent.utils import console


def main() -> None:
    embeddings = get_embeddings()
    console.print(f"[dim]embeddings backend:[/] {type(embeddings).__name__}")

    texts = [
        "LangGraph orchestrates stateful agent workflows.",
        "MCP is a protocol for exposing tools to models.",
    ]
    vectors = embeddings.embed_documents(texts)

    for text, vec in zip(texts, vectors, strict=True):
        console.print(f"\n[bold]{text}[/]")
        console.print(f"  dim = {len(vec)}")
        preview = ", ".join(f"{x:+.3f}" for x in vec[:6])
        console.print(f"  first 6 dims = [{preview}, ...]")


if __name__ == "__main__":
    main()
