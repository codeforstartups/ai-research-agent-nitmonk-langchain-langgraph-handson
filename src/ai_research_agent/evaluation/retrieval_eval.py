"""Retrieval benchmark: measure, don't guess.

We built four retrieval configurations (vector, BM25, hybrid, hybrid+rerank).
Which is best? This module runs each against a labeled QA set and reports
standard IR metrics, so an "improvement" is demonstrated rather than claimed.

Metrics (averaged over questions), given the set of *relevant* source files for
each query and a retriever's top-k results:

* **hit@k**  — did any retrieved chunk come from a relevant source?
* **recall@k** — fraction of relevant sources that were retrieved.
* **MRR**    — 1 / rank of the first relevant chunk (0 if none).

This runs fully offline (in-memory store + local rerank fallback), so the
benchmark is reproducible with no API keys.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever

from ai_research_agent.config import get_settings
from ai_research_agent.ingestion import chunk_documents, load_directory
from ai_research_agent.reranking import Reranker, RerankingRetriever
from ai_research_agent.retrieval import (
    BM25Retriever,
    build_ephemeral_retriever,
    build_hybrid_from_documents,
)

REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_DATASET = REPO_ROOT / "evals" / "datasets" / "retrieval_qa.json"


@dataclass
class RetrievalMetrics:
    hit_rate: float
    recall: float
    mrr: float
    n: int


def load_dataset(path: Path | str = DEFAULT_DATASET) -> list[dict]:
    return json.loads(Path(path).read_text(encoding="utf-8"))["items"]


def _sources(docs: list[Document]) -> list[str]:
    return [d.metadata.get("source", "") for d in docs]


def score_query(retrieved: list[Document], relevant: set[str]) -> tuple[float, float, float]:
    """Return (hit, recall, reciprocal_rank) for one query."""
    got = _sources(retrieved)
    hit = 1.0 if any(s in relevant for s in got) else 0.0
    found = {s for s in got if s in relevant}
    recall = len(found) / len(relevant) if relevant else 0.0
    rr = 0.0
    for rank, source in enumerate(got, start=1):
        if source in relevant:
            rr = 1.0 / rank
            break
    return hit, recall, rr


def evaluate_retriever(
    retriever: BaseRetriever, dataset: list[dict]
) -> RetrievalMetrics:
    hits = recalls = rrs = 0.0
    for item in dataset:
        docs = retriever.invoke(item["question"])
        h, rc, rr = score_query(docs, set(item["relevant_sources"]))
        hits += h
        recalls += rc
        rrs += rr
    n = len(dataset)
    return RetrievalMetrics(hits / n, recalls / n, rrs / n, n)


def build_configurations(chunks: list[Document], *, k: int = 4) -> dict[str, BaseRetriever]:
    """The four retrieval configurations we compare."""
    hybrid = build_hybrid_from_documents(chunks, k=k)
    return {
        "vector": build_ephemeral_retriever(chunks, k=k),
        "bm25": BM25Retriever.from_documents(chunks, k=k),
        "hybrid": hybrid,
        "hybrid+rerank": RerankingRetriever(
            base_retriever=build_hybrid_from_documents(chunks, k=k * 2),
            reranker=Reranker(),
            k=k,
        ),
    }


def run_benchmark(
    *,
    k: int = 4,
    dataset_path: Path | str = DEFAULT_DATASET,
    config_filter: Callable[[str], bool] | None = None,
) -> dict[str, RetrievalMetrics]:
    """Evaluate every retrieval configuration and return per-config metrics."""
    settings = get_settings()
    chunks = chunk_documents(load_directory(settings.knowledge_dir))
    dataset = load_dataset(dataset_path)

    configs = build_configurations(chunks, k=k)
    return {
        name: evaluate_retriever(retriever, dataset)
        for name, retriever in configs.items()
        if config_filter is None or config_filter(name)
    }
