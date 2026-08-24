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
    table.add_row("Vectorstore dir", str(s.vectorstore_dir))
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
