"""Pluggable chat-model factory.

`get_chat_model()` is the *only* place in the codebase that knows how to build a
provider-specific LangChain chat model. Every other module asks for "a chat
model" and stays provider-agnostic — this is what makes the agent portable
across Anthropic, OpenAI and Gemini.

Design:

* The provider is chosen by config (`LLM_PROVIDER`) but can be overridden per
  call — handy when a subagent should use a cheaper/faster model.
* If the selected provider has **no API key**, we fall back to
  :class:`OfflineChatModel` (with a visible warning) so the code path still
  runs. Pass ``allow_offline=False`` to require a real model instead.
"""

from __future__ import annotations

from langchain_core.language_models.chat_models import BaseChatModel

from ai_research_agent.config import LLMProvider, Settings, get_settings
from ai_research_agent.models.offline import OfflineChatModel
from ai_research_agent.utils import console


class MissingCredentialsError(RuntimeError):
    """Raised when a live model is required but the provider key is absent."""


def _build_anthropic(settings: Settings, model: str, temperature: float) -> BaseChatModel:
    from langchain_anthropic import ChatAnthropic

    return ChatAnthropic(
        model=model,  # type: ignore[call-arg]
        temperature=temperature,
        timeout=settings.request_timeout_seconds,
        api_key=settings.anthropic_api_key,  # type: ignore[arg-type]
    )


def _build_openai(settings: Settings, model: str, temperature: float) -> BaseChatModel:
    from langchain_openai import ChatOpenAI

    return ChatOpenAI(
        model=model,
        temperature=temperature,
        timeout=settings.request_timeout_seconds,
        api_key=settings.openai_api_key,  # type: ignore[arg-type]
    )


def _build_google(settings: Settings, model: str, temperature: float) -> BaseChatModel:
    from langchain_google_genai import ChatGoogleGenerativeAI

    return ChatGoogleGenerativeAI(
        model=model,
        temperature=temperature,
        timeout=settings.request_timeout_seconds,
        google_api_key=settings.google_api_key,
    )


def _default_model_name(settings: Settings, provider: LLMProvider) -> str:
    return {
        LLMProvider.ANTHROPIC: settings.anthropic_model,
        LLMProvider.OPENAI: settings.openai_model,
        LLMProvider.GOOGLE: settings.google_model,
    }[provider]


def _has_key(settings: Settings, provider: LLMProvider) -> bool:
    return {
        LLMProvider.ANTHROPIC: settings.has_anthropic,
        LLMProvider.OPENAI: settings.has_openai,
        LLMProvider.GOOGLE: settings.has_google,
    }[provider]


def get_chat_model(
    provider: LLMProvider | None = None,
    *,
    model: str | None = None,
    temperature: float = 0.0,
    allow_offline: bool = True,
    settings: Settings | None = None,
) -> BaseChatModel:
    """Return a chat model for ``provider`` (defaults to the configured one).

    Args:
        provider: override the configured provider.
        model: override the provider's default model name.
        temperature: sampling temperature (0.0 = deterministic, the course
            default for reproducible demos).
        allow_offline: if True and no key is present, return an
            :class:`OfflineChatModel` instead of raising.
        settings: inject settings (tests); defaults to the global singleton.
    """
    settings = settings or get_settings()
    provider = provider or settings.llm_provider
    model_name = model or _default_model_name(settings, provider)

    if not _has_key(settings, provider):
        if not allow_offline:
            raise MissingCredentialsError(
                f"No API key configured for provider '{provider.value}'. "
                f"Set the matching *_API_KEY in .env, or pass allow_offline=True."
            )
        console.print(
            f"[yellow]⚠ No key for '{provider.value}' — using OfflineChatModel "
            f"(deterministic stub). Add the key to .env for real responses.[/]"
        )
        return OfflineChatModel()

    builders = {
        LLMProvider.ANTHROPIC: _build_anthropic,
        LLMProvider.OPENAI: _build_openai,
        LLMProvider.GOOGLE: _build_google,
    }
    return builders[provider](settings, model_name, temperature)
