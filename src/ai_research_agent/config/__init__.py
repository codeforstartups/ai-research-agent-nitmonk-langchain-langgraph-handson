"""Configuration package: typed settings loaded from the environment.

`get_settings()` is the single entry point used everywhere in the codebase so
that configuration is loaded once and shared (see `settings.py`).
"""

from ai_research_agent.config.settings import (
    EmbeddingsProvider,
    LLMProvider,
    Settings,
    get_settings,
)

__all__ = [
    "EmbeddingsProvider",
    "LLMProvider",
    "Settings",
    "get_settings",
]
