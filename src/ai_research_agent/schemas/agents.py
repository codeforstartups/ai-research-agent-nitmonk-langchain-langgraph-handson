"""Schemas for the multi-agent layer."""

from __future__ import annotations

from pydantic import BaseModel, Field


class Finding(BaseModel):
    """A single specialist subagent's contribution."""

    facet: str = Field(description="The angle this finding covers (e.g. 'market').")
    summary: str = Field(description="The specialist's grounded summary.")
    sources: list[str] = Field(default_factory=list, description="Citations used.")


class Verdict(BaseModel):
    """Output of the fact-checker / critic."""

    passed: bool = Field(description="Whether the checked claims are supported.")
    issues: list[str] = Field(default_factory=list, description="Problems found.")
    note: str = Field(default="", description="Short explanation.")
