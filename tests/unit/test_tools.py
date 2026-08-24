"""Phase 11 — LangChain tools."""

from __future__ import annotations

from ai_research_agent.tools import ALL_TOOLS, calculate, percentage_change, search_knowledge


def test_calculate_basic_arithmetic() -> None:
    assert calculate.invoke({"expression": "12000 * 0.15"}) == "1800.0"
    assert calculate.invoke({"expression": "(2 + 3) ** 2"}) == "25.0"


def test_calculate_rejects_unsafe_input() -> None:
    # No eval(): attribute access / calls must be refused, not executed.
    result = calculate.invoke({"expression": "__import__('os').system('echo hi')"})
    assert result.startswith("Error")


def test_percentage_change() -> None:
    assert percentage_change.invoke({"old_value": 100, "new_value": 150}) == "+50.00%"
    assert "undefined" in percentage_change.invoke({"old_value": 0, "new_value": 5})


def test_tool_has_name_and_schema() -> None:
    assert calculate.name == "calculate"
    schema = calculate.args_schema.model_json_schema()
    assert "expression" in schema["properties"]


def test_search_knowledge_returns_cited_text() -> None:
    out = search_knowledge.invoke({"query": "enterprise requirements SOC2"})
    assert "#" in out  # contains a [source#chunk] citation


def test_all_tools_registered() -> None:
    names = {t.name for t in ALL_TOOLS}
    assert names == {"calculate", "percentage_change", "search_knowledge"}
