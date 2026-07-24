from __future__ import annotations

import subprocess

from ducky_agent.tools.base import ToolError


def run_shell(command: str, timeout: float = 10.0) -> str:
    """Run a shell command and return its exit code + combined
    stdout/stderr. Args: command (str), timeout (float, optional, default
    10.0 seconds). The highest-risk tool in this set -- asked-for by
    default under PermissionMode.DEFAULT (see permissions/gate.py), never
    auto-allowed. timeout is a genuine wall-clock kill, the real-subprocess
    analog of grounding.run_sandboxed's SIGALRM discipline (that primitive
    itself is restricted-exec for Python only and can't run shell -- only
    its discipline carries over here).
    """
    try:
        proc = subprocess.run(
            command,
            shell=True,
            timeout=timeout,
            capture_output=True,
            text=True,
        )
    except subprocess.TimeoutExpired:
        raise ToolError(f"command timed out after {timeout}s: {command!r}")

    parts = [f"exit_code: {proc.returncode}"]
    if proc.stdout:
        parts.append(f"stdout:\n{proc.stdout}")
    if proc.stderr:
        parts.append(f"stderr:\n{proc.stderr}")
    return "\n".join(parts)
