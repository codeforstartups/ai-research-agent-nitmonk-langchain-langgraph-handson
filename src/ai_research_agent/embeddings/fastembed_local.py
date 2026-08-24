"""A thin LangChain ``Embeddings`` wrapper around fastembed.

fastembed runs sentence-embedding models locally via ONNX — no API key, no GPU.
The default model (``BAAI/bge-small-en-v1.5``, 384-dim) is small and downloaded
once on first use, then cached. We wrap it (rather than pulling in
``langchain-community``) to keep dependencies lean and to make the
``Embeddings`` interface — ``embed_documents`` / ``embed_query`` — explicit for
teaching.
"""

from __future__ import annotations

from langchain_core.embeddings import Embeddings

DEFAULT_MODEL = "BAAI/bge-small-en-v1.5"


class FastEmbedEmbeddings(Embeddings):
    """Local ONNX embeddings via the ``fastembed`` package."""

    def __init__(self, model_name: str = DEFAULT_MODEL) -> None:
        self.model_name = model_name
        self._model = None  # lazy: don't download until first use

    def _get_model(self):
        if self._model is None:
            from fastembed import TextEmbedding

            self._model = TextEmbedding(model_name=self.model_name)
        return self._model

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        model = self._get_model()
        return [vec.tolist() for vec in model.embed(list(texts))]

    def embed_query(self, text: str) -> list[float]:
        return self.embed_documents([text])[0]
