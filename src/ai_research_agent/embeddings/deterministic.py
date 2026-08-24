"""Deterministic, network-free embeddings.

A hashing bag-of-words embedding: no model download, no API key, fully
reproducible. It is *not* semantically deep, but shared vocabulary raises
cosine similarity — which is exactly enough to teach and test the similarity
mechanics offline and in CI.

For real semantic quality, use fastembed / OpenAI / Google via
:func:`ai_research_agent.embeddings.get_embeddings`.
"""

from __future__ import annotations

import hashlib
import math
import re

from langchain_core.embeddings import Embeddings

_TOKEN = re.compile(r"[a-z0-9]+")


class DeterministicEmbeddings(Embeddings):
    """Hashing bag-of-words embeddings with L2-normalised vectors."""

    def __init__(self, dim: int = 256) -> None:
        self.dim = dim

    def _embed(self, text: str) -> list[float]:
        vec = [0.0] * self.dim
        for token in _TOKEN.findall(text.lower()):
            digest = hashlib.md5(token.encode()).digest()
            index = int.from_bytes(digest[:4], "big") % self.dim
            sign = 1.0 if digest[4] & 1 else -1.0
            vec[index] += sign
        norm = math.sqrt(sum(v * v for v in vec))
        if norm == 0.0:
            return vec
        return [v / norm for v in vec]

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._embed(t) for t in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._embed(text)
