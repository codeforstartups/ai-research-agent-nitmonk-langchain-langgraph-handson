"""Assemble nodes into a compiled LangGraph.

Phase 18 is the simplest possible graph — a straight line:

    START → plan → research → synthesize → END

That's already more than a script: it's an inspectable, visualizable state
machine we can extend with routing, cycles, persistence and human review without
rewriting the nodes.
"""

from __future__ import annotations

from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph

from ai_research_agent.graph.nodes import plan_node, research_node, synthesize_node
from ai_research_agent.graph.state import ResearchState


def build_research_graph() -> CompiledStateGraph:
    """Build and compile the linear research graph."""
    graph = StateGraph(ResearchState)
    graph.add_node("plan", plan_node)
    graph.add_node("research", research_node)
    graph.add_node("synthesize", synthesize_node)

    graph.add_edge(START, "plan")
    graph.add_edge("plan", "research")
    graph.add_edge("research", "synthesize")
    graph.add_edge("synthesize", END)

    return graph.compile()


def run_research(question: str) -> ResearchState:
    """Convenience: run the graph end-to-end for a question."""
    return build_research_graph().invoke({"question": question})
