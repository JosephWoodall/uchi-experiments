"""Shared enums for the permission gate and tool registry. Split out from
gate.py/rules.py (Phase 4) because tools/registry.py (Phase 3) needs
ToolKind to tag each tool -- pure enums, no logic, so building this now
doesn't preempt Phase 4's actual gate/rule-matching behavior.
"""

from __future__ import annotations

import enum


class ToolKind(enum.Enum):
    READ = "read"
    WRITE = "write"
    EXEC = "exec"


class PermissionMode(enum.Enum):
    DEFAULT = "default"
    YOLO = "yolo"
