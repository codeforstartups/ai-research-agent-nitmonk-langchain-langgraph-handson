"""Phase 01 — the chat-model factory and offline fallback."""

from __future__ import annotations

from langchain_core.messages import AIMessageChunk, HumanMessage

from ai_research_agent.config import LLMProvider, Settings
from ai_research_agent.models import OfflineChatModel, get_chat_model
from ai_research_agent.models.chat import MissingCredentialsError


def _settings(**overrides) -> Settings:
    return Settings(_env_file=None, **overrides)  # type: ignore[call-arg]


def test_falls_back_to_offline_when_no_key() -> None:
    model = get_chat_model(LLMProvider.ANTHROPIC, settings=_settings())
    assert isinstance(model, OfflineChatModel)


def test_offline_disallowed_raises() -> None:
    try:
        get_chat_model(LLMProvider.OPENAI, allow_offline=False, settings=_settings())
    except MissingCredentialsError as exc:
        assert "openai" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("expected MissingCredentialsError")


def test_offline_model_invoke_is_deterministic() -> None:
    model = OfflineChatModel()
    out = model.invoke([HumanMessage("Analyze the AI coding agent market")])
    assert "(offline)" in out.text
    assert "AI coding agent market" in out.text


def test_offline_model_streams_chunks() -> None:
    model = OfflineChatModel()
    chunks = list(model.stream([HumanMessage("hello world")]))
    assert len(chunks) > 1
    assert all(isinstance(c, AIMessageChunk) for c in chunks)


def test_real_provider_selected_when_key_present() -> None:
    # A fake key is enough to select the real builder; we don't call the API.
    model = get_chat_model(
        LLMProvider.OPENAI,
        settings=_settings(openai_api_key="sk-test-not-real"),
    )
    assert model.__class__.__name__ == "ChatOpenAI"
