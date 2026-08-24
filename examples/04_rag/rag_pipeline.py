"""Phase 06 — the full RAG pipeline (LCEL).

Run:
    uv run python examples/04_rag/rag_pipeline.py

Teaches:
* LCEL composition: retrieve → format → prompt → generate → project,
* grounding: the answer must come from retrieved evidence and cite it,
* returning both the answer *and* its sources for auditability.
"""

from __future__ import annotations

from ai_research_agent.config import get_settings
from ai_research_agent.ingestion import chunk_documents, load_directory
from ai_research_agent.retrieval import build_ephemeral_retriever, build_rag_chain
from ai_research_agent.utils import console

QUESTION = "Should an enterprise startup enter the AI coding-agent market? Summarise the case."


def main() -> None:
    settings = get_settings()
    chunks = chunk_documents(load_directory(settings.knowledge_dir))
    retriever = build_ephemeral_retriever(chunks, k=4)

    chain = build_rag_chain(retriever)
    result = chain.invoke(QUESTION)

    console.rule("[bold]Question[/]")
    console.print(QUESTION)
    console.rule("[bold]Grounded answer[/]")
    console.print(result["answer"])
    console.rule("[bold]Sources[/]")
    for doc in result["sources"]:
        console.print(f"  - {doc.metadata['source']}#{doc.metadata['chunk']}")


if __name__ == "__main__":
    main()
