"""Phase 04 — load and chunk the knowledge base.

Run:
    uv run python examples/04_rag/ingest_and_chunk.py

Teaches:
* loading heterogeneous sources into a common ``Document`` type,
* why we chunk (embeddable units + traceable citations),
* how metadata (``source``, ``chunk``) rides along for later attribution.
"""

from __future__ import annotations

from ai_research_agent.config import get_settings
from ai_research_agent.ingestion import chunk_documents, load_directory
from ai_research_agent.utils import console


def main() -> None:
    settings = get_settings()
    console.print(f"[dim]knowledge dir:[/] {settings.knowledge_dir}")

    documents = load_directory(settings.knowledge_dir)
    console.print(f"Loaded [bold]{len(documents)}[/] source documents.")

    chunks = chunk_documents(documents)
    console.print(f"Split into [bold]{len(chunks)}[/] chunks.\n")

    console.rule("[bold]First 3 chunks[/]")
    for chunk in chunks[:3]:
        meta = chunk.metadata
        console.print(f"[cyan]{meta['source']}#{meta['chunk']}[/] ({len(chunk.page_content)} chars)")
        preview = chunk.page_content[:180].replace("\n", " ")
        console.print(f"  {preview}…\n")


if __name__ == "__main__":
    main()
