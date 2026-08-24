"""Ingestion package: load raw files and split them into chunks."""

from ai_research_agent.ingestion.chunking import chunk_documents
from ai_research_agent.ingestion.loaders import load_directory, load_file

__all__ = ["chunk_documents", "load_directory", "load_file"]
