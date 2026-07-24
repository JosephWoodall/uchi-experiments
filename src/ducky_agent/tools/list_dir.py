from __future__ import annotations

from pathlib import Path

from ducky_agent.tools.base import ToolError


def list_dir(path: str = ".") -> str:
    """List entries in a directory, one per line, directories suffixed
    with '/'. Args: path (str, default '.')."""
    p = Path(path)
    if not p.exists():
        raise ToolError(f"no such directory: {path}")
    if not p.is_dir():
        raise ToolError(f"not a directory: {path}")
    entries = sorted(p.iterdir(), key=lambda e: e.name)
    lines = [f"{e.name}/" if e.is_dir() else e.name for e in entries]
    return "\n".join(lines) if lines else "(empty directory)"
