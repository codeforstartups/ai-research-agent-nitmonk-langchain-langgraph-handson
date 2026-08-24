"""Phase 02 — structured output.

Run:
    uv run python examples/01_llm/structured_output.py

Teaches:
* `.with_structured_output(Schema)` — make the model return a *typed object*,
  not a string you have to parse,
* why typed outputs are the backbone of a reliable agent (validation, safe
  hand-off between nodes),
* the same call working offline (deterministic heuristic plan).
"""

from __future__ import annotations

from langchain_core.messages import HumanMessage, SystemMessage

from ai_research_agent.models import get_chat_model, get_structured_response
from ai_research_agent.schemas import ResearchPlan
from ai_research_agent.utils import console

QUESTION = (
    "Analyze the AI coding-agent market and determine whether an enterprise "
    "startup should enter this market."
)


def main() -> None:
    model = get_chat_model()
    messages = [
        SystemMessage(
            "You are a research planner. Decompose the user's question into "
            "3–6 focused sub-questions. Mark which need live web data."
        ),
        HumanMessage(QUESTION),
    ]

    plan = get_structured_response(model, ResearchPlan, messages)

    console.rule("[bold]Typed ResearchPlan[/]")
    console.print(f"[dim]type:[/] {type(plan).__name__}  •  validated by Pydantic")
    for line in plan.as_lines():
        console.print(line)


if __name__ == "__main__":
    main()
