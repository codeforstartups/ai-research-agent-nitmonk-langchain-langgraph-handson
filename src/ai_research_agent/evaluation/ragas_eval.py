"""Ragas evaluation of the RAG pipeline.

Retrieval metrics (retrieval_eval.py) tell us *what was retrieved*. Ragas tells
us about the *generated answer* using an LLM judge:

* **faithfulness** — is every claim in the answer supported by the retrieved
  context (i.e. no hallucination)?
* **answer/response relevancy** — does the answer actually address the question?
* **context precision / recall (with reference)** — was the retrieved context
  on-target and sufficient, measured against reference answers?

These metrics require a real LLM judge (and embeddings), so this path is only
available when an LLM key is configured. It runs against the same labeled
dataset as the retrieval benchmark.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from ai_research_agent.config import get_settings
from ai_research_agent.evaluation.retrieval_eval import DEFAULT_DATASET, load_dataset


class RagasUnavailableError(RuntimeError):
    """Raised when Ragas evaluation is requested without an LLM key."""


def _any_llm_key() -> bool:
    s = get_settings()
    return s.has_anthropic or s.has_openai or s.has_google


def run_ragas_eval(
    *,
    k: int = 4,
    dataset_path: Path | str = DEFAULT_DATASET,
) -> Any:
    """Run Ragas metrics over the RAG pipeline; return the EvaluationResult.

    Raises :class:`RagasUnavailableError` if no LLM key is configured.
    """
    if not _any_llm_key():
        raise RagasUnavailableError(
            "Ragas needs an LLM judge. Set ANTHROPIC_API_KEY / OPENAI_API_KEY / "
            "GOOGLE_API_KEY in .env to run answer-quality evaluation."
        )

    # Heavy + version-sensitive imports are kept local (and shimmed first).
    from ai_research_agent.evaluation import _compat

    _compat.install()

    from ragas import EvaluationDataset, evaluate
    from ragas.dataset_schema import SingleTurnSample
    from ragas.embeddings import LangchainEmbeddingsWrapper
    from ragas.llms import LangchainLLMWrapper
    from ragas.metrics import (
        Faithfulness,
        LLMContextPrecisionWithReference,
        LLMContextRecall,
        ResponseRelevancy,
    )

    from ai_research_agent.embeddings import get_embeddings
    from ai_research_agent.ingestion import chunk_documents, load_directory
    from ai_research_agent.models import get_chat_model
    from ai_research_agent.retrieval import build_ephemeral_retriever, build_rag_chain

    settings = get_settings()
    chunks = chunk_documents(load_directory(settings.knowledge_dir))
    retriever = build_ephemeral_retriever(chunks, k=k)
    rag_chain = build_rag_chain(retriever)

    samples = []
    for item in load_dataset(dataset_path):
        result = rag_chain.invoke(item["question"])
        samples.append(
            SingleTurnSample(
                user_input=item["question"],
                response=result["answer"],
                retrieved_contexts=[d.page_content for d in result["sources"]],
                reference=item["ground_truth"],
            )
        )

    evaluation_dataset = EvaluationDataset(samples=samples)
    judge = LangchainLLMWrapper(get_chat_model(allow_offline=False))
    embedder = LangchainEmbeddingsWrapper(get_embeddings())

    return evaluate(
        dataset=evaluation_dataset,
        metrics=[
            Faithfulness(),
            ResponseRelevancy(),
            LLMContextPrecisionWithReference(),
            LLMContextRecall(),
        ],
        llm=judge,
        embeddings=embedder,
    )
