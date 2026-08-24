"""Schemas for the research workflow.

These are deliberately introduced early (phase-02) because a *typed plan* is the
first artifact the agent produces — the Planner node in the final LangGraph
emits exactly this :class:`ResearchPlan`.
"""

from __future__ import annotations

from pydantic import BaseModel, Field


class SubQuestion(BaseModel):
    """One decomposed piece of the overall research question."""

    question: str = Field(description="A specific, answerable sub-question.")
    rationale: str = Field(
        description="Why answering this sub-question helps address the objective."
    )
    needs_web: bool = Field(
        default=True,
        description="Whether current/live web information is likely required.",
    )


class ResearchPlan(BaseModel):
    """A structured decomposition of a complex research question."""

    objective: str = Field(description="A one-sentence restatement of the goal.")
    sub_questions: list[SubQuestion] = Field(
        description="3–6 sub-questions that together cover the objective.",
        min_length=1,
    )

    def as_lines(self) -> list[str]:
        """Human-readable rendering used by the CLI/demos."""
        lines = [f"Objective: {self.objective}", ""]
        for i, sq in enumerate(self.sub_questions, 1):
            tag = "web" if sq.needs_web else "internal"
            lines.append(f"{i}. ({tag}) {sq.question}")
            lines.append(f"     ↳ {sq.rationale}")
        return lines
