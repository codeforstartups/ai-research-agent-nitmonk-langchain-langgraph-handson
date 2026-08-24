"""Cross-encoder reranking.

Retrieval (dense/sparse/hybrid) is fast but shallow: it scores query and
document *independently*. A **cross-encoder** reranker reads the query and each
candidate *together*, so it judges true relevance far more accurately — at a
cost, which is why we only rerank a shortlist of candidates, not the whole
corpus.

We use **FlashRank** (small, local, ONNX cross-encoders). If it can't load
(offline/CI), we fall back to a deterministic lexical-overlap score so the
mechanics and tests still run — clearly a downgrade, not a silent one.
"""

from __future__ import annotations

from langchain_core.documents import Document

from ai_research_agent.retrieval.bm25 import tokenize
from ai_research_agent.utils import console


def _lexical_scores(query: str, docs: list[Document]) -> list[float]:
    """Fallback relevance: Jaccard-ish overlap of query/document tokens."""
    q = set(tokenize(query))
    scores = []
    for doc in docs:
        d = set(tokenize(doc.page_content))
        scores.append(len(q & d) / len(q) if q else 0.0)
    return scores


class Reranker:
    """Rerank candidate documents against a query."""

    def __init__(self, *, use_flashrank: bool = True, model_name: str | None = None) -> None:
        self.use_flashrank = use_flashrank
        self.model_name = model_name
        self._ranker = None

    def _get_ranker(self):
        if self._ranker is None:
            from flashrank import Ranker

            self._ranker = Ranker(model_name=self.model_name) if self.model_name else Ranker()
        return self._ranker

    def rerank(self, query: str, docs: list[Document], *, top_n: int = 4) -> list[Document]:
        if not docs:
            return []
        if self.use_flashrank:
            try:
                return self._rerank_flashrank(query, docs, top_n)
            except Exception as exc:  # pragma: no cover - depends on env
                console.print(
                    f"[yellow]⚠ flashrank unavailable ({exc.__class__.__name__}); "
                    f"using lexical fallback reranker.[/]"
                )
        scores = _lexical_scores(query, docs)
        order = sorted(range(len(docs)), key=lambda i: scores[i], reverse=True)
        return [docs[i] for i in order[:top_n]]

    def _rerank_flashrank(self, query: str, docs: list[Document], top_n: int) -> list[Document]:
        from flashrank import RerankRequest

        passages = [{"id": i, "text": d.page_content} for i, d in enumerate(docs)]
        results = self._get_ranker().rerank(RerankRequest(query=query, passages=passages))
        # FlashRank returns passages sorted by descending score.
        return [docs[r["id"]] for r in results[:top_n]]
