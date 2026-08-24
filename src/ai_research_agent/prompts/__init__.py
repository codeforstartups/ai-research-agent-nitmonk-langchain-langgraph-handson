"""Prompt templates, kept in one place so they're easy to inspect and tune.

Prompts are part of the trust boundary: the RAG system prompt explicitly frames
retrieved context as *evidence to cite*, never as instructions to follow — the
same discipline we later extend to untrusted web content.
"""

from ai_research_agent.prompts.rag import RAG_PROMPT, format_docs

__all__ = ["RAG_PROMPT", "format_docs"]
