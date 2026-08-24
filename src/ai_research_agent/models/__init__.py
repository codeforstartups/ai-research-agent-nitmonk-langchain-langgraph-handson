"""Model factories (chat models now; embeddings arrive in phase-03)."""

from ai_research_agent.models.chat import MissingCredentialsError, get_chat_model
from ai_research_agent.models.offline import OfflineChatModel

__all__ = ["MissingCredentialsError", "OfflineChatModel", "get_chat_model"]
