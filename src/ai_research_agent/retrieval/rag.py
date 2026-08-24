"""The RAG pipeline, composed with LCEL.

RAG = Retrieval-Augmented Generation: retrieve relevant chunks, stuff them into
a prompt as evidence, and let the model answer *grounded* in that evidence
(instead of from parametric memory). This is the whole reason we built ingestion
+ embeddings + a vector store.

We compose it with **LCEL** (the LangChain Expression Language) so the data flow
is explicit and every stage is independently runnable/inspectable:

    {docs, question} → +context → +answer → {answer, sources}
"""

from __future__ import annotations

from typing import Any, TypedDict

from langchain_core.documents import Document
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import (
    Runnable,
    RunnableLambda,
    RunnableParallel,
    RunnablePassthrough,
)
from langchain_core.vectorstores import VectorStoreRetriever

from ai_research_agent.models import get_chat_model
from ai_research_agent.prompts import RAG_PROMPT, format_docs
from ai_research_agent.retrieval.vectorstore import get_retriever, load_vectorstore


class RagResult(TypedDict):
    answer: str
    sources: list[Document]


def build_rag_chain(
    retriever: VectorStoreRetriever,
    model: BaseChatModel | None = None,
) -> Runnable[str, RagResult]:
    """Build a RAG chain: ``question: str`` → ``{answer, sources}``."""
    model = model or get_chat_model()

    # Stage 1: fetch docs and carry the original question forward.
    fetch = RunnableParallel(docs=retriever, question=RunnablePassthrough())

    # Stage 2: format the docs into a citeable context string.
    add_context = RunnablePassthrough.assign(
        context=lambda x: format_docs(x["docs"])
    )

    # Stage 3: generate the grounded answer from context + question.
    generate: Runnable[dict[str, Any], str] = RAG_PROMPT | model | StrOutputParser()
    add_answer = RunnablePassthrough.assign(answer=generate)

    # Stage 4: project to the public result shape.
    project = RunnableLambda(
        lambda x: RagResult(answer=x["answer"], sources=x["docs"])
    )

    return fetch | add_context | add_answer | project


def answer_question(
    question: str,
    *,
    k: int = 4,
    model: BaseChatModel | None = None,
) -> RagResult:
    """Convenience: open the persisted store and answer a question."""
    retriever = get_retriever(load_vectorstore(), k=k)
    chain = build_rag_chain(retriever, model=model)
    return chain.invoke(question)
