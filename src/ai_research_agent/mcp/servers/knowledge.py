"""Knowledge MCP server — internal RAG exposed over the protocol.

The Web Research MCP brings *live external* knowledge; this server brings
*persistent internal* knowledge. Exposing RAG as an MCP server (rather than a
hard-wired pipeline stage) means the agent chooses when to consult internal docs
using the same tool-calling mechanism it uses for the web — a uniform interface
over two very different context sources.

Primitives:
* **tools**: ``search_knowledge``, ``get_document``, ``list_sources``,
* **resource**: ``knowledge://document/{source}`` (full document by name),
* **prompt**: ``summarize_source``.
"""

from __future__ import annotations

from functools import lru_cache

from mcp.server import MCPServer

from ai_research_agent.config import get_settings
from ai_research_agent.ingestion import chunk_documents, load_directory
from ai_research_agent.prompts import format_docs
from ai_research_agent.retrieval import build_ephemeral_retriever


@lru_cache(maxsize=1)
def _load():
    """Load raw documents once; return (docs_by_source, retriever)."""
    settings = get_settings()
    documents = load_directory(settings.knowledge_dir)
    by_source: dict[str, str] = {}
    for doc in documents:
        src = doc.metadata["source"]
        by_source[src] = by_source.get(src, "") + doc.page_content
    retriever = build_ephemeral_retriever(chunk_documents(documents), k=4)
    return by_source, retriever


def build_knowledge_server() -> MCPServer:
    """Construct the Knowledge MCP server."""
    server: MCPServer = MCPServer(
        name="knowledge",
        version="0.1.0",
        instructions="Search and retrieve the internal knowledge base (RAG over MCP).",
    )

    @server.tool(description="Semantic search over internal documents; returns cited passages.")
    def search_knowledge(query: str, k: int = 4) -> str:
        _, retriever = _load()
        docs = retriever.invoke(query)[:k]
        return format_docs(docs) if docs else "No relevant internal documents found."

    @server.tool(description="List the available internal source documents.")
    def list_sources() -> str:
        by_source, _ = _load()
        return "\n".join(sorted(by_source)) or "No documents indexed."

    @server.tool(description="Return the full text of a named internal document.")
    def get_document(source: str) -> str:
        by_source, _ = _load()
        return by_source.get(source, f"No document named {source!r}.")

    @server.resource(
        "knowledge://document/{source}",
        description="Full text of a single internal document by filename.",
        mime_type="text/plain",
    )
    def document_resource(source: str) -> str:
        by_source, _ = _load()
        return by_source.get(source, f"No document named {source!r}.")

    @server.prompt(description="Template to summarise a named internal document.")
    def summarize_source(source: str) -> str:
        return (
            f"Read the internal document '{source}' (via get_document) and summarise "
            f"its key claims in 3-5 bullets, noting anything a decision-maker must know."
        )

    return server
