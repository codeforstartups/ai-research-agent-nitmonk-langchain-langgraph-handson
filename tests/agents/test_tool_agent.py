"""Phase 17 — the tool-calling agent loop.

We drive the loop with a *scripted* model so the test is deterministic: it
requests a tool on the first turn, then answers on the second.
"""

from __future__ import annotations

from typing import Any

from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.outputs import ChatGeneration, ChatResult

from ai_research_agent.agents import run_tool_agent
from ai_research_agent.models import OfflineChatModel
from ai_research_agent.tools import calculate


class ScriptedModel(BaseChatModel):
    """Yields a queued sequence of AIMessages, ignoring inputs."""

    responses: list[AIMessage]

    @property
    def _llm_type(self) -> str:
        return "scripted"

    def bind_tools(self, tools, **kwargs):  # noqa: ANN001, ANN003
        return self

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> ChatResult:
        message = self.responses.pop(0)
        return ChatResult(generations=[ChatGeneration(message=message)])


async def test_agent_executes_tool_then_answers() -> None:
    scripted = ScriptedModel(
        responses=[
            AIMessage(
                content="",
                tool_calls=[{"name": "calculate", "args": {"expression": "6 * 7"}, "id": "c1"}],
            ),
            AIMessage(content="The answer is 42."),
        ]
    )
    result = await run_tool_agent("What is 6*7?", [calculate], model=scripted)
    assert result.answer == "The answer is 42."
    assert len(result.steps) == 1
    assert result.steps[0].tool == "calculate"
    assert result.steps[0].output == "42.0"


async def test_agent_handles_unknown_tool_gracefully() -> None:
    scripted = ScriptedModel(
        responses=[
            AIMessage(
                content="",
                tool_calls=[{"name": "nope", "args": {}, "id": "c1"}],
            ),
            AIMessage(content="Done."),
        ]
    )
    result = await run_tool_agent("x", [calculate], model=scripted)
    assert "unknown tool" in result.steps[0].output


async def test_agent_respects_max_iterations() -> None:
    # Always asks for a tool -> never terminates -> hits the cap.
    def _loop_msg() -> AIMessage:
        return AIMessage(
            content="",
            tool_calls=[{"name": "calculate", "args": {"expression": "1+1"}, "id": "c"}],
        )

    scripted = ScriptedModel(responses=[_loop_msg() for _ in range(10)])
    result = await run_tool_agent("x", [calculate], model=scripted, max_iterations=3)
    assert result.iterations == 3
    assert "maximum" in result.answer


async def test_offline_model_answers_without_tools() -> None:
    result = await run_tool_agent("Summarise the market", [calculate], model=OfflineChatModel())
    assert result.steps == []
    assert "(offline)" in result.answer
