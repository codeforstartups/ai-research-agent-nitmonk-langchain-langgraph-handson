"""Specialist research subagents.

Rather than one generalist doing everything, we use focused specialists — each
with a **defined responsibility**, a **narrow retrieval focus**, and a typed
**output schema** (:class:`Finding`). A supervisor runs them in parallel and an
aggregator combines their findings.

Why bother? Focused agents produce sharper, more auditable evidence than one
prompt trying to cover market + tech + competition at once — and each can be
evaluated on its own facet.
"""

from __future__ import annotations

from collections.abc import Callable
from functools import lru_cache

from langchain_core.messages import HumanMessage, SystemMessage

from ai_research_agent.config import get_settings
from ai_research_agent.graph.state import ResearchState
from ai_research_agent.ingestion import chunk_documents, load_directory
from ai_research_agent.models import get_chat_model
from ai_research_agent.retrieval import build_ephemeral_retriever
from ai_research_agent.schemas.agents import Finding


@lru_cache(maxsize=1)
def _retriever():
    chunks = chunk_documents(load_directory(get_settings().knowledge_dir))
    return build_ephemeral_retriever(chunks, k=3)


def make_specialist(
    facet: str,
    focus_query: str,
    system_prompt: str,
) -> Callable[[ResearchState], ResearchState]:
    """Build a specialist node that researches one ``facet``.

    The node retrieves evidence for its focus, summarises it within its remit,
    and returns exactly one :class:`Finding` (accumulated via the state reducer).
    """

    def specialist(state: ResearchState) -> ResearchState:
        retriever = _retriever()
        query = f"{focus_query} {state['question']}"
        docs = retriever.invoke(query)
        sources = [f"{d.metadata['source']}#{d.metadata['chunk']}" for d in docs]
        context = "\n".join(
            f"[{d.metadata['source']}#{d.metadata['chunk']}] {d.page_content}" for d in docs
        )
        model = get_chat_model()
        messages = [
            SystemMessage(system_prompt),
            HumanMessage(f"Question: {state['question']}\n\nEvidence:\n{context}"),
        ]
        summary = model.invoke(messages).text
        finding = Finding(facet=facet, summary=summary, sources=sources)
        return {"findings": [finding]}

    specialist.__name__ = f"{facet}_specialist"
    return specialist


market_specialist = make_specialist(
    "market",
    "market size segmentation demand drivers",
    "You are a market analyst. Cover ONLY market structure, segments and demand. "
    "Ground every claim in the evidence and cite [source#chunk].",
)

competitor_specialist = make_specialist(
    "competitor",
    "competitors players moats weaknesses",
    "You are a competitive analyst. Cover ONLY competitors, their moats and "
    "weaknesses. Ground every claim in the evidence and cite [source#chunk].",
)

enterprise_specialist = make_specialist(
    "enterprise",
    "enterprise requirements security compliance governance",
    "You are an enterprise-readiness analyst. Cover ONLY buyer requirements, "
    "security, compliance and governance. Cite [source#chunk].",
)

ALL_SPECIALISTS = [market_specialist, competitor_specialist, enterprise_specialist]
