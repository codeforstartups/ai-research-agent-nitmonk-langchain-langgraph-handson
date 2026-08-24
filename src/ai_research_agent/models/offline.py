"""A deterministic offline chat model.

The course must run end-to-end *without* API keys — in CI, on a plane, or
before the viewer has signed up for a provider. :class:`OfflineChatModel` is a
tiny, honest stand-in: it does no reasoning, it simply echoes structured
acknowledgements of the conversation so that all the *plumbing* (message
construction, streaming, LCEL wiring) can be exercised and demonstrated.

It is intentionally obvious that it is a stub — the point on camera is
"everything is wired; add a key and real intelligence flows through the exact
same code path."
"""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any

from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain_core.language_models.chat_models import SimpleChatModel
from langchain_core.messages import BaseMessage, HumanMessage


class OfflineChatModel(SimpleChatModel):
    """A no-network chat model that deterministically echoes its input."""

    # Not bracketed: Rich would parse "[offline]" as a markup tag and hide it.
    marker: str = "(offline)"

    @property
    def _llm_type(self) -> str:
        return "offline-echo"

    def _last_human_text(self, messages: list[BaseMessage]) -> str:
        for message in reversed(messages):
            if isinstance(message, HumanMessage):
                return message.text
        return messages[-1].text if messages else ""

    def _call(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> str:
        prompt = self._last_human_text(messages).strip()
        preview = (prompt[:160] + "…") if len(prompt) > 160 else prompt
        return (
            f"{self.marker} No live model configured, so this is a deterministic "
            f"stub response. You asked: “{preview}”. Set a provider API key in "
            f".env to route this exact call to a real model."
        )

    def _stream(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> Iterator[Any]:
        # Emit token-by-token so streaming demos look real even offline.
        from langchain_core.messages import AIMessageChunk
        from langchain_core.outputs import ChatGenerationChunk

        text = self._call(messages, stop=stop, run_manager=run_manager, **kwargs)
        for token in text.split(" "):
            chunk = ChatGenerationChunk(message=AIMessageChunk(content=token + " "))
            if run_manager:
                run_manager.on_llm_new_token(token, chunk=chunk)
            yield chunk
