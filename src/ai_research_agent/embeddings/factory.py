"""Pluggable embeddings factory — the retrieval-side twin of `get_chat_model`.

`get_embeddings()` returns a LangChain ``Embeddings`` for the configured
provider:

* ``fastembed`` — local ONNX, no key (course default, zero cost),
* ``openai`` — ``text-embedding-3-*`` (needs ``OPENAI_API_KEY``),
* ``google`` — Gemini embeddings (needs ``GOOGLE_API_KEY``).

If a keyed provider is selected without a key, we fall back to local fastembed
(and if even that can't load, to :class:`DeterministicEmbeddings`) so retrieval
phases always run.
"""

from __future__ import annotations

from langchain_core.embeddings import Embeddings

from ai_research_agent.config import EmbeddingsProvider, Settings, get_settings
from ai_research_agent.embeddings.deterministic import DeterministicEmbeddings
from ai_research_agent.embeddings.fastembed_local import FastEmbedEmbeddings
from ai_research_agent.utils import console


def _fastembed_or_deterministic() -> Embeddings:
    try:
        model = FastEmbedEmbeddings()
        model.embed_query("warmup")  # force the (cached) model load now
        return model
    except Exception as exc:  # pragma: no cover - depends on network/cache
        console.print(
            f"[yellow]⚠ fastembed unavailable ({exc.__class__.__name__}); "
            f"using DeterministicEmbeddings.[/]"
        )
        return DeterministicEmbeddings()


def get_embeddings(
    provider: EmbeddingsProvider | None = None,
    *,
    settings: Settings | None = None,
) -> Embeddings:
    """Return an embeddings model for ``provider`` (defaults to configured)."""
    settings = settings or get_settings()
    provider = provider or settings.embeddings_provider

    if provider is EmbeddingsProvider.OPENAI:
        if not settings.has_openai:
            console.print("[yellow]⚠ No OPENAI_API_KEY — falling back to fastembed.[/]")
            return _fastembed_or_deterministic()
        from langchain_openai import OpenAIEmbeddings

        return OpenAIEmbeddings(
            model="text-embedding-3-small",
            api_key=settings.openai_api_key,  # type: ignore[arg-type]
        )

    if provider is EmbeddingsProvider.GOOGLE:
        if not settings.has_google:
            console.print("[yellow]⚠ No GOOGLE_API_KEY — falling back to fastembed.[/]")
            return _fastembed_or_deterministic()
        from langchain_google_genai import GoogleGenerativeAIEmbeddings

        return GoogleGenerativeAIEmbeddings(
            model="models/text-embedding-004",
            google_api_key=settings.google_api_key,
        )

    # Default: local fastembed.
    return _fastembed_or_deterministic()
