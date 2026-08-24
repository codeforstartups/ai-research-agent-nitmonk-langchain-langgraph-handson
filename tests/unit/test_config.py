"""Phase 00 — configuration loads and exposes provider selection."""

from __future__ import annotations

from ai_research_agent.config import (
    EmbeddingsProvider,
    LLMProvider,
    Settings,
    get_settings,
)


def test_settings_defaults() -> None:
    """With no env vars set, sensible defaults are used."""
    s = Settings(_env_file=None)  # type: ignore[call-arg]
    assert s.llm_provider is LLMProvider.ANTHROPIC
    assert s.embeddings_provider is EmbeddingsProvider.FASTEMBED
    assert s.request_timeout_seconds == 60


def test_provider_enums_are_string_valued() -> None:
    """Enums are str-backed so they serialise cleanly to env/JSON."""
    assert LLMProvider.OPENAI.value == "openai"
    assert EmbeddingsProvider.GOOGLE.value == "google"


def test_service_availability_flags_default_false() -> None:
    s = Settings(_env_file=None)  # type: ignore[call-arg]
    assert s.has_anthropic is False
    assert s.has_thordata is False
    assert s.langfuse_active is False


def test_get_settings_is_cached() -> None:
    get_settings.cache_clear()
    assert get_settings() is get_settings()
