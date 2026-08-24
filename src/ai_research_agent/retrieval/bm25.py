"""BM25 sparse (keyword) retrieval.

Vector search matches by *meaning* but can miss exact terms — product names,
acronyms, error codes, rare jargon. BM25 is the classic keyword ranking
function: it scores documents by term overlap with the query, weighting rare
terms higher and saturating on repetition. It complements dense retrieval, which
is exactly why we fuse the two in phase 08.

We implement a small `BaseRetriever` so BM25 plugs into the same
``.invoke(query)`` interface as the vector retriever.
"""

from __future__ import annotations

import re
from typing import Any

from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from pydantic import ConfigDict
from rank_bm25 import BM25Okapi

_TOKEN = re.compile(r"[a-z0-9]+")


def tokenize(text: str) -> list[str]:
    return _TOKEN.findall(text.lower())


class BM25Retriever(BaseRetriever):
    """A keyword retriever backed by ``rank_bm25.BM25Okapi``."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    documents: list[Document]
    bm25: Any
    k: int = 4

    @classmethod
    def from_documents(cls, documents: list[Document], *, k: int = 4) -> BM25Retriever:
        corpus = [tokenize(doc.page_content) for doc in documents]
        # BM25Okapi requires a non-empty corpus.
        bm25 = BM25Okapi(corpus if corpus else [[""]])
        return cls(documents=documents, bm25=bm25, k=k)

    def _get_relevant_documents(
        self, query: str, *, run_manager: CallbackManagerForRetrieverRun
    ) -> list[Document]:
        if not self.documents:
            return []
        scores = self.bm25.get_scores(tokenize(query))
        ranked = sorted(
            zip(self.documents, scores, strict=True),
            key=lambda pair: pair[1],
            reverse=True,
        )
        return [doc for doc, score in ranked[: self.k] if score > 0]
