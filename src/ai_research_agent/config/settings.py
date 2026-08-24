"""Typed application settings.

We use `pydantic-settings` so that every configuration value is:

* declared once, with a type and a default,
* loaded from the process environment (and a local ``.env`` file),
* validated at startup rather than failing deep inside a request.

The provider fields (:class:`LLMProvider`, :class:`EmbeddingsProvider`) are the
seed of the *pluggable* design: later phases build model factories that switch
on these enums so the same agent can run on Anthropic, OpenAI or Gemini without
touching business logic.
"""

from __future__ import annotations

from enum import Enum
from functools import lru_cache
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

# Repository root = three parents up from this file:
#   src/ai_research_agent/config/settings.py -> repo root
REPO_ROOT = Path(__file__).resolve().parents[3]


class LLMProvider(str, Enum):
    """Supported chat-model providers."""

    ANTHROPIC = "anthropic"
    OPENAI = "openai"
    GOOGLE = "google"


class EmbeddingsProvider(str, Enum):
    """Supported embeddings providers.

    ``fastembed`` runs locally (ONNX) with no API key, which keeps the RAG
    phases zero-cost and reproducible on any machine.
    """

    FASTEMBED = "fastembed"
    OPENAI = "openai"
    GOOGLE = "google"


class Settings(BaseSettings):
    """All runtime configuration, sourced from the environment / ``.env``."""

    model_config = SettingsConfigDict(
        env_file=REPO_ROOT / ".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    # ---- Provider selection -------------------------------------------------
    llm_provider: LLMProvider = LLMProvider.ANTHROPIC
    embeddings_provider: EmbeddingsProvider = EmbeddingsProvider.FASTEMBED

    # ---- Model names --------------------------------------------------------
    anthropic_model: str = "claude-sonnet-5"
    openai_model: str = "gpt-4.1"
    google_model: str = "gemini-2.5-pro"

    # ---- Provider API keys --------------------------------------------------
    anthropic_api_key: str | None = None
    openai_api_key: str | None = None
    google_api_key: str | None = None

    # ---- Thordata -----------------------------------------------------------
    thordata_api_key: str | None = None
    thordata_base_url: str = "https://api.thordata.com"
    thordata_timeout_seconds: int = 30

    # ---- Langfuse -----------------------------------------------------------
    langfuse_public_key: str | None = None
    langfuse_secret_key: str | None = None
    langfuse_host: str = "https://cloud.langfuse.com"
    langfuse_enabled: bool = True

    # ---- Vector store (Pinecone, serverless) --------------------------------
    pinecone_api_key: str | None = None
    pinecone_index: str = "ai-research-agent"
    pinecone_cloud: str = "aws"
    pinecone_region: str = "us-east-1"

    # ---- Retrieval paths ----------------------------------------------------
    knowledge_dir: Path = Field(default=REPO_ROOT / "data" / "knowledge")

    # ---- Runtime ------------------------------------------------------------
    request_timeout_seconds: int = 60
    log_level: str = "INFO"

    # ---- Convenience flags --------------------------------------------------
    @property
    def has_anthropic(self) -> bool:
        return bool(self.anthropic_api_key)

    @property
    def has_openai(self) -> bool:
        return bool(self.openai_api_key)

    @property
    def has_google(self) -> bool:
        return bool(self.google_api_key)

    @property
    def has_thordata(self) -> bool:
        return bool(self.thordata_api_key)

    @property
    def has_pinecone(self) -> bool:
        return bool(self.pinecone_api_key)

    @property
    def langfuse_active(self) -> bool:
        """Langfuse is only wired up when enabled *and* fully credentialed."""
        return (
            self.langfuse_enabled
            and bool(self.langfuse_public_key)
            and bool(self.langfuse_secret_key)
        )


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return the process-wide :class:`Settings` singleton.

    Cached so the ``.env`` file is parsed once. Call ``get_settings.cache_clear()``
    in tests that need to reload configuration.
    """
    return Settings()
