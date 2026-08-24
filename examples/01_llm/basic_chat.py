"""Phase 01 — talking to a model through LangChain.

Run:
    uv run python examples/01_llm/basic_chat.py

Teaches:
* the message types (System / Human / AI) and why chat models take a *list*,
* provider-agnostic model construction via `get_chat_model()`,
* `.invoke()` for a single response and `.stream()` for token streaming.

Works with no API key (falls back to the offline stub) — set a provider key in
`.env` to see real model output through the identical code path.
"""

from __future__ import annotations

from langchain_core.messages import HumanMessage, SystemMessage

from ai_research_agent.models import get_chat_model
from ai_research_agent.utils import console


def main() -> None:
    model = get_chat_model()

    messages = [
        SystemMessage(
            "You are a precise research analyst. Answer in two crisp sentences."
        ),
        HumanMessage("What is the Model Context Protocol (MCP), in one line?"),
    ]

    console.rule("[bold]invoke()[/] — single response")
    response = model.invoke(messages)
    console.print(response.text)

    console.rule("[bold]stream()[/] — token by token")
    for chunk in model.stream(messages):
        console.print(chunk.text, end="")
    console.print()  # newline after the stream


if __name__ == "__main__":
    main()
