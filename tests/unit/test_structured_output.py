"""Phase 02 — structured output helper (offline paths)."""

from __future__ import annotations

from langchain_core.messages import HumanMessage
from pydantic import BaseModel

from ai_research_agent.models import OfflineChatModel, get_structured_response
from ai_research_agent.schemas import ResearchPlan


def test_offline_research_plan_is_valid_and_populated() -> None:
    model = OfflineChatModel()
    plan = get_structured_response(
        model, ResearchPlan, [HumanMessage("Analyze the AI coding agent market")]
    )
    assert isinstance(plan, ResearchPlan)
    assert len(plan.sub_questions) >= 1
    assert any(sq.needs_web for sq in plan.sub_questions)
    assert "AI coding agent market" in plan.objective


class _Nested(BaseModel):
    label: str
    score: float


class _Sample(BaseModel):
    title: str
    count: int
    flag: bool
    tags: list[str]
    nested: _Nested
    note: str | None


def test_generic_placeholder_builds_arbitrary_schema() -> None:
    model = OfflineChatModel()
    obj = get_structured_response(model, _Sample, [HumanMessage("x")])
    assert isinstance(obj, _Sample)
    assert obj.count == 0
    assert obj.flag is False
    assert obj.tags and isinstance(obj.tags[0], str)
    assert isinstance(obj.nested, _Nested)
    assert obj.note is None
