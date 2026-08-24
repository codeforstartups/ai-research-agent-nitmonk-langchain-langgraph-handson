"""Analysis tools — safe, deterministic helpers the model can call.

Two lessons here:

1. **Tools are model-controlled actions** with a *typed* signature. LangChain
   turns the function signature + docstring into a schema the model uses to
   decide when and how to call the tool.
2. **A tool is authority.** ``calculate`` must never ``eval()`` arbitrary input
   — we parse a restricted arithmetic AST instead. This is the same discipline
   we later apply to MCP tools and untrusted web content.
"""

from __future__ import annotations

import ast
import operator

from langchain_core.tools import tool

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


@tool
def calculate(expression: str) -> str:
    """Evaluate a basic arithmetic expression (+, -, *, /, %, ** and parentheses).

    Use for exact arithmetic on numbers the model would otherwise guess at.
    Example: "market of 12000 * 0.15".
    """
    try:
        tree = ast.parse(expression, mode="eval")
        return str(_safe_eval(tree.body))
    except (ValueError, SyntaxError, ZeroDivisionError) as exc:
        return f"Error: {exc}"


@tool
def percentage_change(old_value: float, new_value: float) -> str:
    """Compute the percentage change from old_value to new_value.

    Use to quantify growth/decline between two metrics.
    """
    if old_value == 0:
        return "Error: percentage change from zero is undefined."
    pct = (new_value - old_value) / abs(old_value) * 100.0
    return f"{pct:+.2f}%"
