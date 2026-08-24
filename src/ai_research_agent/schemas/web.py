"""Schemas for web research results."""

from __future__ import annotations

from pydantic import BaseModel, Field


class SearchResult(BaseModel):
    """One normalized web search hit."""

    title: str = Field(description="Result title.")
    url: str = Field(description="Canonical URL of the result (preserved verbatim).")
    snippet: str = Field(default="", description="Short summary/snippet.")
    position: int = Field(default=0, description="Rank in the result list (1-based).")
    source: str = Field(default="thordata", description="Where the result came from.")


class ScrapedPage(BaseModel):
    """The extracted content of a single URL."""

    url: str
    title: str = ""
    content: str = ""
    source: str = "thordata"
