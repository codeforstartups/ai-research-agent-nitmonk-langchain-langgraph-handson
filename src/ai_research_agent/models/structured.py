"""Structured (typed) model outputs.

`get_structured_response()` asks a model to return an instance of a Pydantic
schema instead of free text. With a real provider this uses LangChain's
``.with_structured_output(schema)`` (tool/JSON-mode under the hood). Offline, it
produces a deterministic instance so the mechanics can still be demonstrated:

* schemas with a registered *offline builder* get a sensible heuristic result,
* any other schema gets a clearly-labelled placeholder instance.
"""

from __future__ import annotations

import types
import typing
from collections.abc import Callable
from typing import TypeVar

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import BaseMessage, HumanMessage
from pydantic import BaseModel

from ai_research_agent.models.offline import OfflineChatModel
from ai_research_agent.schemas.research import ResearchPlan, SubQuestion

T = TypeVar("T", bound=BaseModel)

_PLACEHOLDER = "offline placeholder — set an API key for a real value"


def _last_human_text(messages: list[BaseMessage]) -> str:
    for message in reversed(messages):
        if isinstance(message, HumanMessage):
            return message.text
    return messages[-1].text if messages else ""


def _offline_research_plan(prompt: str) -> ResearchPlan:
    """A deterministic, obviously-heuristic plan for offline demos."""
    objective = prompt.strip() or "Investigate the requested topic."
    return ResearchPlan(
        objective=f"Determine an answer to: {objective}",
        sub_questions=[
            SubQuestion(
                question=f"What is the current state of the market/topic in: {objective}?",
                rationale="Establish an up-to-date baseline before analysis.",
                needs_web=True,
            ),
            SubQuestion(
                question="Who are the key players/competitors and how do they differ?",
                rationale="Map the competitive landscape.",
                needs_web=True,
            ),
            SubQuestion(
                question="What internal knowledge, risks, and constraints are relevant?",
                rationale="Combine live findings with persistent internal context.",
                needs_web=False,
            ),
        ],
    )


# Schemas we can answer usefully even without a model.
_OFFLINE_BUILDERS: dict[type[BaseModel], Callable[[str], BaseModel]] = {
    ResearchPlan: _offline_research_plan,
}


def _placeholder_value(annotation: object) -> object:
    """Build a type-appropriate placeholder for an arbitrary annotation."""
    origin = typing.get_origin(annotation)

    # Optional[X] / X | None -> None
    if origin in (typing.Union, types.UnionType):
        args = [a for a in typing.get_args(annotation) if a is not type(None)]
        return None if len(args) < len(typing.get_args(annotation)) else _placeholder_value(args[0])

    if origin in (list, set, tuple):
        (inner,) = typing.get_args(annotation) or (str,)
        return [_placeholder_value(inner)]
    if origin is dict:
        return {}

    if isinstance(annotation, type):
        if issubclass(annotation, BaseModel):
            return _build_placeholder(annotation)
        if issubclass(annotation, bool):
            return False
        if issubclass(annotation, int):
            return 0
        if issubclass(annotation, float):
            return 0.0
    return _PLACEHOLDER


def _build_placeholder(schema: type[T]) -> T:
    values = {
        name: _placeholder_value(field.annotation)
        for name, field in schema.model_fields.items()
    }
    return schema(**values)


def get_structured_response(
    model: BaseChatModel,
    schema: type[T],
    messages: list[BaseMessage],
) -> T:
    """Return an instance of ``schema`` produced by ``model``."""
    if isinstance(model, OfflineChatModel):
        prompt = _last_human_text(messages)
        builder = _OFFLINE_BUILDERS.get(schema)
        if builder is not None:
            return builder(prompt)  # type: ignore[return-value]
        return _build_placeholder(schema)

    structured = model.with_structured_output(schema)
    return structured.invoke(messages)  # type: ignore[return-value]
