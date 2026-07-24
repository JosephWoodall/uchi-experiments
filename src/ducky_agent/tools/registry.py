"""The tool registry: name -> (fn, ToolKind, description). The one place
that maps a parsed action's tool name to something executable, and the one
place ToolError becomes the uniform ToolResult contract."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from ducky_agent.permissions.types import ToolKind
from ducky_agent.tools.base import ToolError, ToolResult, truncate_output
from ducky_agent.tools.list_dir import list_dir
from ducky_agent.tools.read_file import read_file
from ducky_agent.tools.run_shell import run_shell
from ducky_agent.tools.write_file import write_file


@dataclass
class ToolSpec:
    fn: Callable[..., str]
    kind: ToolKind
    description: str


TOOL_REGISTRY: dict[str, ToolSpec] = {
    "read_file": ToolSpec(
        read_file, ToolKind.READ, "Read a file's full text content. Args: path (str)."
    ),
    "list_dir": ToolSpec(
        list_dir, ToolKind.READ, "List entries in a directory. Args: path (str, default '.')."
    ),
    "write_file": ToolSpec(
        write_file,
        ToolKind.WRITE,
        "Overwrite a file with new content (whole-file only). Args: path (str), content (str).",
    ),
    "run_shell": ToolSpec(
        run_shell,
        ToolKind.EXEC,
        "Run a shell command. Args: command (str), timeout (float, optional).",
    ),
}


def known_tool_names() -> frozenset[str]:
    """The action parser's ``known_tools`` argument -- kept as a function,
    not a module-level constant, so it always reflects the live registry
    even if tools are added/removed at runtime (tests do this)."""
    return frozenset(TOOL_REGISTRY)


def execute_tool(name: str, args: dict, max_output_chars: int = 2000) -> ToolResult:
    spec = TOOL_REGISTRY.get(name)
    if spec is None:
        return ToolResult(ok=False, output=f"unknown tool: {name}")
    try:
        raw_output = spec.fn(**args)
    except ToolError as e:
        return ToolResult(ok=False, output=str(e))
    except TypeError as e:
        # Wrong/missing/extra args -- the model's own call didn't match the
        # tool's real signature; report it the same way as any other
        # expected tool failure, not an unhandled crash.
        return ToolResult(ok=False, output=f"invalid arguments for {name}: {e}")
    truncated_output, was_truncated = truncate_output(raw_output, max_output_chars)
    return ToolResult(ok=True, output=truncated_output, truncated=was_truncated)
