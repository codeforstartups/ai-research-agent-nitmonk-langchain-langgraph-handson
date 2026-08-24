"""Phase 10 — benchmark the retrieval stack (offline, no keys).

Run:
    uv run python examples/11_evaluation/retrieval_benchmark.py

Teaches:
* turning "hybrid+rerank feels better" into measured hit@k / recall@k / MRR,
* the BUILD → RUN → EVALUATE loop: we now have numbers to improve against.
"""

from __future__ import annotations

from ai_research_agent.evaluation import run_benchmark
from ai_research_agent.utils import console


def main() -> None:
    console.rule("[bold]Retrieval configurations, measured[/]")
    results = run_benchmark(k=4)

    header = f"{'config':<16}{'hit@k':>8}{'recall@k':>10}{'MRR':>8}"
    console.print(header)
    console.print("-" * len(header))
    for name, m in results.items():
        console.print(f"{name:<16}{m.hit_rate:>8.2f}{m.recall:>10.2f}{m.mrr:>8.2f}")

    best = max(results.items(), key=lambda kv: (kv[1].mrr, kv[1].recall))
    console.print(f"\nBest by MRR: [bold green]{best[0]}[/]")


if __name__ == "__main__":
    main()
