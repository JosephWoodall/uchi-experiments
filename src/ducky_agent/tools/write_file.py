from __future__ import annotations

from pathlib import Path

from ducky_agent.tools.base import ToolError


def write_file(path: str, content: str) -> str:
    """Overwrite a file with new content (whole-file only -- no diff/patch
    application; see the plan's explicit cut of edit_file). Args: path
    (str), content (str). The parent directory must already exist -- this
    tool deliberately does not silently create directory structure the
    caller didn't ask for.
    """
    p = Path(path)
    if not p.parent.exists():
        raise ToolError(f"parent directory does not exist: {p.parent}")
    if p.exists() and p.is_dir():
        raise ToolError(f"path is a directory, not a file: {path}")
    p.write_text(content)
    return f"wrote {len(content)} chars to {path}"
