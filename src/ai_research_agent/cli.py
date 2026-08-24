"""Command-line interface for the research agent.

Phase 00 ships a single ``doctor`` command that prints the resolved
configuration and shows which live services are wired up. Later phases add
``ingest``, ``retrieve``, ``ask`` and finally the full ``run`` command.
"""

from __future__ import annotations

import typer
from rich.panel import Panel
from rich.table import Table

from ai_research_agent import __version__
from ai_research_agent.config import get_settings
from ai_research_agent.utils import console

app = typer.Typer(
    name="research-agent",
    help="AI Research & Analysis Agent — phase-by-phase teaching CLI.",
    no_args_is_help=True,
    add_completion=False,
)


def _require_pinecone_or_exit() -> None:
    """Print a friendly message and exit if the Pinecone key is missing.

    The product retrieval path (ingest/search/rag) uses Pinecone, which needs a
    key. This keeps the CLI from dumping a raw traceback on camera.
    """
    if not get_settings().has_pinecone:
        console.print(
            "[yellow]Pinecone is not configured.[/] The product retrieval "
            "commands (ingest/search/rag) need [cyan]PINECONE_API_KEY[/] in .env.\n"
            "For a no-key demo of the same concepts, run the examples in "
            "[cyan]examples/04_rag/[/]."
        )
        raise typer.Exit(code=1)


@app.command()
def version() -> None:
    """Print the package version."""
    console.print(f"ai-research-agent [bold cyan]v{__version__}[/]")


@app.command()
def ask(
    prompt: str = typer.Argument(..., help="A question to send to the model."),
    stream: bool = typer.Option(False, "--stream", help="Stream tokens as they arrive."),
) -> None:
    """Send a single prompt to the configured chat model (Phase 01)."""
    from langchain_core.messages import HumanMessage, SystemMessage

    from ai_research_agent.models import get_chat_model

    model = get_chat_model()
    messages = [
        SystemMessage("You are a concise, precise research analyst."),
        HumanMessage(prompt),
    ]

    if stream:
        for chunk in model.stream(messages):
            console.print(chunk.text, end="")
        console.print()
    else:
        response = model.invoke(messages)
        console.print(response.text)


@app.command()
def plan(
    question: str = typer.Argument(..., help="A complex research question to decompose."),
) -> None:
    """Produce a typed ResearchPlan from a question (Phase 02)."""
    from langchain_core.messages import HumanMessage, SystemMessage

    from ai_research_agent.models import get_chat_model, get_structured_response
    from ai_research_agent.schemas import ResearchPlan

    model = get_chat_model()
    messages = [
        SystemMessage(
            "You are a research planner. Decompose the question into 3–6 focused "
            "sub-questions and mark which need live web data."
        ),
        HumanMessage(question),
    ]
    result = get_structured_response(model, ResearchPlan, messages)
    for line in result.as_lines():
        console.print(line)


@app.command()
def ingest() -> None:
    """Load, chunk, and index the knowledge base into the vector store (Phase 05)."""
    from ai_research_agent.retrieval import index_knowledge_base

    _require_pinecone_or_exit()
    settings = get_settings()
    num_docs, num_chunks = index_knowledge_base()
    console.print(
        f"Indexed [bold]{num_docs}[/] documents → [bold]{num_chunks}[/] chunks into "
        f"Pinecone index [cyan]{settings.pinecone_index}[/] ✓"
    )


@app.command()
def search(
    query: str = typer.Argument(..., help="Query to run against the vector store."),
    k: int = typer.Option(4, "--k", help="Number of chunks to return."),
) -> None:
    """Semantic search over the indexed knowledge base (Phase 05)."""
    from ai_research_agent.retrieval import get_retriever, load_vectorstore

    _require_pinecone_or_exit()
    store = load_vectorstore()
    retriever = get_retriever(store, k=k)
    results = retriever.invoke(query)
    if not results:
        console.print("[yellow]No results. Did you run 'research-agent ingest'?[/]")
        return
    for i, doc in enumerate(results, 1):
        meta = doc.metadata
        cite = f"{meta.get('source', '?')}#{meta.get('chunk', '?')}"
        console.print(f"[bold]{i}. [cyan]{cite}[/][/]")
        console.print(f"   {doc.page_content[:200].strip()}…\n")


@app.command()
def rag(
    question: str = typer.Argument(..., help="Question to answer from the knowledge base."),
    k: int = typer.Option(4, "--k", help="Chunks to retrieve as evidence."),
) -> None:
    """Answer a question with grounded, cited RAG (Phase 06)."""
    from ai_research_agent.retrieval import answer_question

    _require_pinecone_or_exit()
    result = answer_question(question, k=k)
    console.rule("[bold]Answer[/]")
    console.print(result["answer"])
    console.rule("[bold]Sources[/]")
    for doc in result["sources"]:
        console.print(f"  - {doc.metadata.get('source')}#{doc.metadata.get('chunk')}")


@app.command()
def benchmark(k: int = typer.Option(4, "--k", help="Top-k for each retriever.")) -> None:
    """Benchmark the 4 retrieval configs against the labeled set (Phase 10, offline)."""
    from ai_research_agent.evaluation import run_benchmark

    results = run_benchmark(k=k)
    table = Table(title=f"Retrieval benchmark (k={k})", header_style="bold magenta")
    table.add_column("Config", style="cyan")
    table.add_column("hit@k", justify="right")
    table.add_column("recall@k", justify="right")
    table.add_column("MRR", justify="right")
    for name, m in results.items():
        table.add_row(name, f"{m.hit_rate:.2f}", f"{m.recall:.2f}", f"{m.mrr:.2f}")
    console.print(table)


@app.command()
def evaluate(k: int = typer.Option(4, "--k", help="Top-k retrieved as context.")) -> None:
    """Run Ragas answer-quality evaluation (Phase 10, needs an LLM key)."""
    from ai_research_agent.evaluation import RagasUnavailableError, run_ragas_eval

    try:
        result = run_ragas_eval(k=k)
    except RagasUnavailableError as exc:
        console.print(f"[yellow]{exc}[/]")
        raise typer.Exit(code=1) from None
    console.rule("[bold]Ragas scores[/]")
    console.print(result)


@app.command()
def doctor() -> None:
    """Show resolved configuration and which live services are available.

    This is the 'is my environment set up?' command demonstrated on camera at
    the end of Phase 00.
    """
    s = get_settings()

    table = Table(title="Configuration", show_header=True, header_style="bold magenta")
    table.add_column("Setting", style="cyan", no_wrap=True)
    table.add_column("Value", style="white")

    table.add_row("LLM provider", s.llm_provider.value)
    table.add_row("Embeddings provider", s.embeddings_provider.value)
    table.add_row("Anthropic model", s.anthropic_model)
    table.add_row("OpenAI model", s.openai_model)
    table.add_row("Google model", s.google_model)
    table.add_row("Pinecone index", s.pinecone_index)
    table.add_row("Knowledge dir", str(s.knowledge_dir))
    console.print(table)

    services = Table(title="Live services", show_header=True, header_style="bold magenta")
    services.add_column("Service", style="cyan")
    services.add_column("Status")

    def mark(ok: bool) -> str:
        return "[green]● ready[/]" if ok else "[yellow]○ not configured (using fallback)[/]"

    services.add_row("Anthropic", mark(s.has_anthropic))
    services.add_row("OpenAI", mark(s.has_openai))
    services.add_row("Google", mark(s.has_google))
    services.add_row("Pinecone", mark(s.has_pinecone))
    services.add_row("Thordata", mark(s.has_thordata))
    services.add_row("Langfuse", mark(s.langfuse_active))
    console.print(services)

    console.print(
        Panel.fit(
            "Environment loaded successfully. Copy [cyan].env.example[/] to "
            "[cyan].env[/] and add keys to light up live services.",
            title="Phase 00 ✓",
            border_style="green",
        )
    )


if __name__ == "__main__":
    app()
