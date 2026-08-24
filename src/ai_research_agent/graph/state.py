"""The graph's shared state.

LangGraph passes a single typed **state** object between nodes. Each node
returns a *partial* update; LangGraph merges it in. Fields annotated with a
**reducer** (like ``operator.add`` on a list) accumulate across nodes instead of
overwriting — which is what lets parallel research branches all contribute
evidence (phase 19).
"""

from __future__ import annotations

import operator
from typing import Annotated, TypedDict

from ai_research_agent.schemas.research import ResearchPlan


class ResearchState(TypedDict, total=False):
    """State threaded through the research graph."""

    question: str
    plan: ResearchPlan
    # `add` reducer: every node that returns evidence appends to the list.
    evidence: Annotated[list[str], operator.add]
    sources: Annotated[list[str], operator.add]
    answer: str
