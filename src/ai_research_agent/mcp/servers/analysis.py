"""Analysis MCP server — the foundational example server.

It demonstrates all three MCP primitives in one small, offline server:

* **tools**: ``calculate`` and ``percentage_change`` (model-controlled actions),
* **resource**: ``analysis://methods`` (application-controlled context — a
  catalogue of what this server can do),
* **prompt**: ``compare_competitors`` (a user-controlled reusable template).

Run it over stdio (`build_analysis_server().run_stdio_async()`), inspect it with
the MCP Inspector, or connect a client in-memory in tests.
"""

from __future__ import annotations

import ast
import operator

from mcp.server import MCPServer

_ALLOWED_BINOPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.Pow: operator.pow,
    ast.Mod: operator.mod,
}
_ALLOWED_UNARY = {ast.UAdd: operator.pos, ast.USub: operator.neg}


def _safe_eval(node: ast.AST) -> float:
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return float(node.value)
    if isinstance(node, ast.BinOp) and type(node.op) in _ALLOWED_BINOPS:
        return _ALLOWED_BINOPS[type(node.op)](_safe_eval(node.left), _safe_eval(node.right))
    if isinstance(node, ast.UnaryOp) and type(node.op) in _ALLOWED_UNARY:
        return _ALLOWED_UNARY[type(node.op)](_safe_eval(node.operand))
    raise ValueError("Unsupported or unsafe expression")


def build_analysis_server() -> MCPServer:
    """Construct and return the Analysis MCP server."""
    server: MCPServer = MCPServer(
        name="analysis",
        version="0.1.0",
        instructions="Deterministic analysis tools: arithmetic and metric comparison.",
    )

    @server.tool(description="Evaluate a basic arithmetic expression (+ - * / % ** and parens).")
    def calculate(expression: str) -> str:
        try:
            return str(_safe_eval(ast.parse(expression, mode="eval").body))
        except (ValueError, SyntaxError, ZeroDivisionError) as exc:
            return f"Error: {exc}"

    @server.tool(description="Percentage change from old_value to new_value.")
    def percentage_change(old_value: float, new_value: float) -> str:
        if old_value == 0:
            return "Error: percentage change from zero is undefined."
        return f"{(new_value - old_value) / abs(old_value) * 100.0:+.2f}%"

    @server.resource(
        "analysis://methods",
        description="Catalogue of analysis methods this server exposes.",
        mime_type="text/plain",
    )
    def methods() -> str:
        return (
            "analysis methods:\n"
            "- calculate(expression): safe arithmetic\n"
            "- percentage_change(old_value, new_value): growth/decline %"
        )

    @server.prompt(description="Template for a competitor comparison analysis.")
    def compare_competitors(company_a: str, company_b: str, dimension: str) -> str:
        return (
            f"Compare {company_a} and {company_b} on '{dimension}'. "
            f"Use the analysis tools for any quantitative claims, cite evidence, "
            f"and end with a one-line recommendation."
        )

    return server
