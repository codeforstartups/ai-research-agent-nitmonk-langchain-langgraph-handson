"""A LangGraph whose nodes call MCP capabilities.

Phases 18-20 retrieved via in-process helpers. Here the research node instead
goes through the **MCP client** — so the graph gathers evidence from whatever
servers are connected (web + knowledge), with no knowledge of *how* they work.
That's the modularity argument for MCP: to give the graph a new capability you
register a server, you don't touch the graph.

Nodes that call MCP are **async** (MCP is async), so run the graph with
``await graph.ainvoke(...)`` inside an open ``MultiMCPClient`` context.
"""

from __future__ import annotations

from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph

from ai_research_agent.graph.nodes import plan_node, synthesize_node
from ai_research_agent.graph.state import ResearchState
from ai_research_agent.mcp.client import MultiMCPClient


def build_mcp_graph(client: MultiMCPClient) -> CompiledStateGraph:
    """Build a graph whose research node gathers evidence via MCP tools.

    ``client`` must be an *open* :class:`MultiMCPClient` (inside its
    ``async with``) for the lifetime of the graph run.
    """

    async def mcp_research_node(state: ResearchState) -> ResearchState:
        plan = state.get("plan")
        sub_questions = (
            [sq.question for sq in plan.sub_questions] if plan else [state["question"]]
        )
        tools = {t.qualified_name for t in client.list_tools()}

        evidence: list[str] = []
        sources: list[str] = []
        for sub in sub_questions[:3]:
            if "knowledge__search_knowledge" in tools:
                out = await client.call_tool("knowledge__search_knowledge", {"query": sub})
                evidence.append(f"[knowledge] {out}")
                sources.append("knowledge-mcp")
            if "web_research__search_web" in tools:
                out = await client.call_tool("web_research__search_web", {"query": sub, "count": 2})
                evidence.append(f"[web] {out}")
                sources.append("web-mcp")
        return {"evidence": evidence, "sources": sources}

    graph = StateGraph(ResearchState)
    graph.add_node("plan", plan_node)
    graph.add_node("mcp_research", mcp_research_node)
    graph.add_node("synthesize", synthesize_node)

    graph.add_edge(START, "plan")
    graph.add_edge("plan", "mcp_research")
    graph.add_edge("mcp_research", "synthesize")
    graph.add_edge("synthesize", END)

    return graph.compile()
