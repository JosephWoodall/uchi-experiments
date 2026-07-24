"""Shared tool-result contract. Individual tool functions (read_file.py etc.)
are kept simple: return a plain string on success, raise ToolError on an
expected failure (file not found, command timeout, ...). registry.py's
execute_tool() is the one place that catches ToolError and turns it into a
uniform ToolResult -- tool implementations never construct ToolResult
themselves.
"""

from __future__ import annotations

from dataclasses import dataclass


class ToolError(Exception):
    """An expected, reportable tool failure -- caught by execute_tool() and
    turned into ToolResult(ok=False, ...), never allowed to propagate as an
    unhandled exception up through the agent loop."""


@dataclass
class ToolResult:
    ok: bool
    output: str
    truncated: bool = False


def truncate_output(text: str, max_chars: int = 2000) -> tuple[str, bool]:
    """A 128-token real context window (the DEFAULT_RUN checkpoint's
    block_size, see tasks/ducky.md) makes unbounded tool output actively
    harmful, not just inefficient -- one large file read or verbose shell
    command could otherwise consume the entire prompt budget. Applied
    uniformly to every tool's output in execute_tool(), not cherry-picked
    per tool, since any tool can in principle return something long (a
    directory with thousands of entries is exactly as unbounded as a large
    file).
    """
    if len(text) <= max_chars:
        return text, False
    return text[:max_chars] + f"\n...[truncated, {len(text) - max_chars} more chars]", True
