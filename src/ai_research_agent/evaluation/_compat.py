"""Compatibility shim for importing ragas alongside langchain-core v1.

ragas 0.4.x hard-imports ``langchain_community.chat_models.vertexai.ChatVertexAI``
at module load. That submodule was removed in langchain-community 0.4 (the
version required by langchain-core v1), so ``import ragas`` raises
``ModuleNotFoundError`` even though we never use VertexAI.

We register a minimal stub for that one submodule *before* ragas is imported.
Call :func:`install` at the top of any module that imports ragas.
"""

from __future__ import annotations

import sys
import types

_MODULE = "langchain_community.chat_models.vertexai"


def install() -> None:
    if _MODULE in sys.modules:
        return
    try:  # prefer the real class if the standalone package is installed
        from langchain_google_vertexai import ChatVertexAI  # type: ignore
    except Exception:

        class ChatVertexAI:  # minimal placeholder; ragas only needs the symbol
            """Stub — VertexAI is not used in this project."""

    stub = types.ModuleType(_MODULE)
    stub.ChatVertexAI = ChatVertexAI  # type: ignore[attr-defined]
    sys.modules[_MODULE] = stub
