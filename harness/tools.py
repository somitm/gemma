"""Tools — the actions the model can ask the harness to run.

A tool is just a function plus a JSON-schema contract. The registry turns those
functions into OpenAI tool specs (so the model knows what it can call) and
dispatches the calls by name, parsing arguments and returning a string result —
or an error string the model can read and recover from.

Tools are an API surface you expose to a model: keep the list small, keep each
contract narrow, and validate arguments. ``calculator`` evaluates arithmetic
without ``eval``; ``read_file`` returns a file's contents. (``read_file`` is
unscoped here — it can read any path. Confining tools to a workspace is the
execution-environment concern of ch-08.)
"""

from __future__ import annotations

import ast
import json
import operator
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

_BINOPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.Pow: operator.pow,
    ast.Mod: operator.mod,
}


def calculator(expression: str) -> str:
    """Evaluate a basic arithmetic expression safely (no eval, just numbers + + - * / % **)."""

    def ev(node: ast.AST) -> float:
        if isinstance(node, ast.Constant) and isinstance(node.value, int | float):
            return node.value
        if isinstance(node, ast.BinOp) and type(node.op) in _BINOPS:
            return _BINOPS[type(node.op)](ev(node.left), ev(node.right))
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
            return -ev(node.operand)
        raise ValueError("unsupported expression")

    result = ev(ast.parse(expression, mode="eval").body)
    return str(int(result) if result == int(result) else result)


def read_file(path: str) -> str:
    """Return a file's contents, or an error string."""
    p = Path(path)
    return p.read_text() if p.is_file() else f"error: no such file: {path}"


@dataclass
class Tool:
    name: str
    description: str
    parameters: dict
    func: Callable[..., str]


class ToolRegistry:
    def __init__(self) -> None:
        self._tools: dict[str, Tool] = {}

    def register(self, tool: Tool) -> None:
        self._tools[tool.name] = tool

    def specs(self) -> list[dict]:
        return [
            {
                "type": "function",
                "function": {
                    "name": t.name,
                    "description": t.description,
                    "parameters": t.parameters,
                },
            }
            for t in self._tools.values()
        ]

    def call(self, name: str, arguments: str) -> str:
        tool = self._tools.get(name)
        if tool is None:
            return f"error: unknown tool {name!r}"
        try:
            args = json.loads(arguments) if arguments else {}
        except json.JSONDecodeError:
            return f"error: could not parse arguments {arguments!r}"
        try:
            return str(tool.func(**args))
        except Exception as exc:  # noqa: BLE001 — tool errors are fed back to the model
            return f"error: {exc}"

    def __len__(self) -> int:
        return len(self._tools)


def default_tools() -> ToolRegistry:
    reg = ToolRegistry()
    reg.register(
        Tool(
            name="calculator",
            description="Evaluate an arithmetic expression like '47 * 89'.",
            parameters={
                "type": "object",
                "properties": {"expression": {"type": "string"}},
                "required": ["expression"],
            },
            func=calculator,
        )
    )
    reg.register(
        Tool(
            name="read_file",
            description="Read a UTF-8 text file from disk and return its contents.",
            parameters={
                "type": "object",
                "properties": {"path": {"type": "string"}},
                "required": ["path"],
            },
            func=read_file,
        )
    )
    return reg
