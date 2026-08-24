"""Vector similarity helpers.

Cosine similarity is the workhorse of semantic search: it measures the *angle*
between two embedding vectors, ignoring their magnitude. 1.0 = identical
direction, 0.0 = orthogonal (unrelated).
"""

from __future__ import annotations

import numpy as np


def cosine_similarity(a: list[float], b: list[float]) -> float:
    """Cosine similarity between two vectors, in ``[-1, 1]``."""
    va, vb = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    na, nb = np.linalg.norm(va), np.linalg.norm(vb)
    if na == 0.0 or nb == 0.0:
        return 0.0
    return float(np.dot(va, vb) / (na * nb))


def rank_by_similarity(
    query_vec: list[float],
    doc_vecs: list[list[float]],
    top_k: int | None = None,
) -> list[tuple[int, float]]:
    """Rank document vectors against a query vector.

    Returns ``(index, score)`` pairs sorted by descending similarity.
    """
    scored = [(i, cosine_similarity(query_vec, dv)) for i, dv in enumerate(doc_vecs)]
    scored.sort(key=lambda pair: pair[1], reverse=True)
    return scored if top_k is None else scored[:top_k]
