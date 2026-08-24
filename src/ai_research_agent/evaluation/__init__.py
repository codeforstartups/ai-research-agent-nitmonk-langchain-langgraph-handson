"""Evaluation package: retrieval benchmark (offline) + Ragas answer quality."""

from ai_research_agent.evaluation.ragas_eval import RagasUnavailableError, run_ragas_eval
from ai_research_agent.evaluation.retrieval_eval import (
    RetrievalMetrics,
    evaluate_retriever,
    run_benchmark,
)

__all__ = [
    "RagasUnavailableError",
    "RetrievalMetrics",
    "evaluate_retriever",
    "run_benchmark",
    "run_ragas_eval",
]
