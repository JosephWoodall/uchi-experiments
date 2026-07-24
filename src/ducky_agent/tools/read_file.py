from __future__ import annotations

from pathlib import Path

from ducky_agent.tools.base import ToolError


def read_file(path: str) -> str:
    """Read a file's full text content. Args: path (str)."""
    p = Path(path)
    if not p.exists():
        raise ToolError(f"no such file: {path}")
    if not p.is_file():
        raise ToolError(f"not a file: {path}")
    try:
        return p.read_text()
    except UnicodeDecodeError:
        raise ToolError(f"cannot read as text (binary file?): {path}")
