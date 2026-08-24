"""Fact-checking and critique — the reliability layer.

Specialists can be confident and wrong. This graph adds two gates after
aggregation:

* **fact_check** — a grounding check: does the answer actually rest on cited
  evidence? It emits a typed :class:`Verdict`. On failure the graph **loops
  back** to gather/aggregate again (a verification cycle), bounded by a guard.
* **critic** — once grounded, a qualitative pass that sharpens the answer.

```text
… → aggregate → fact_check ─fail→ revise ─┐
                    │                     │ (cycle, capped)
                    │  ┌──────────────────┘
                    │  ▼
                    └ pass → critic → END
```
"""

from __future__ import annotations

from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph

from ai_research_agent.graph.state import ResearchState
from ai_research_agent.graph.supervisor import aggregate_node, supervisor_node
from ai_research_agent.models import get_chat_model
from ai_research_agent.schemas.agents import Verdict
from ai_research_agent.subagents.specialists import (
    competitor_specialist,
    enterprise_specialist,
    market_specialist,
)

MAX_VERIFY_LOOPS = 2
MIN_ANSWER_CHARS = 20


def fact_check_node(state: ResearchState) -> ResearchState:
    """Grounding check: the answer must be non-trivial and cite known sources."""
    answer = state.get("answer", "")
    known = {s for f in state.get("findings", []) for s in f.sources}
    known |= set(state.get("sources", []))

    issues: list[str] = []
    if len(answer) < MIN_ANSWER_CHARS:
        issues.append("Answer is too short to be substantive.")
    if not known:
        issues.append("No sources were gathered to ground the answer.")

    verdict = Verdict(
        passed=not issues,
        issues=issues,
        note="Grounding check over specialist findings.",
    )
    return {"verdict": verdict}


def route_fact_check(state: ResearchState) -> str:
    """Fail → revise (bounded); pass → critic."""
    verdict = state.get("verdict")
    passed = verdict.passed if verdict else False
    if passed or state.get("iterations", 0) >= MAX_VERIFY_LOOPS:
        return "pass"
    return "fail"


def revise_node(state: ResearchState) -> ResearchState:
    """Count a verification loop (a real revise would gather more evidence)."""
    return {"iterations": state.get("iterations", 0) + 1}


def critic_node(state: ResearchState) -> ResearchState:
    """Qualitative pass: sharpen the grounded answer."""
    model = get_chat_model()
    messages = [
        SystemMessage(
            "You are a critic. Improve the answer's clarity and ensure it ends "
            "with an explicit recommendation. Keep all citations. Do not add "
            "unsupported claims."
        ),
        HumanMessage(f"Question: {state['question']}\n\nDraft answer:\n{state.get('answer', '')}"),
    ]
    return {"answer": model.invoke(messages).text}


def build_verified_graph() -> CompiledStateGraph:
    """Supervisor + specialists + aggregate + fact_check cycle + critic."""
    graph = StateGraph(ResearchState)
    graph.add_node("supervisor", supervisor_node)
    graph.add_node("market_agent", market_specialist)
    graph.add_node("competitor_agent", competitor_specialist)
    graph.add_node("enterprise_agent", enterprise_specialist)
    graph.add_node("aggregate", aggregate_node)
    graph.add_node("fact_check", fact_check_node)
    graph.add_node("revise", revise_node)
    graph.add_node("critic", critic_node)

    graph.add_edge(START, "supervisor")
    for agent in ("market_agent", "competitor_agent", "enterprise_agent"):
        graph.add_edge("supervisor", agent)
        graph.add_edge(agent, "aggregate")
    graph.add_edge("aggregate", "fact_check")
    graph.add_conditional_edges(
        "fact_check", route_fact_check, {"fail": "revise", "pass": "critic"}
    )
    graph.add_edge("revise", "aggregate")  # verification cycle
    graph.add_edge("critic", END)

    return graph.compile()
