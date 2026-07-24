"""The permission gate: allow/deny/ask policy, nothing more. Modeled on the
reference harness's own separation (its permissions/gate.py never prompts
either) -- harness/session.py owns the actual human interaction, which
keeps this class pure logic and lets ScriptedModel-based tests drive it
deterministically with a scripted on_ask callback instead of real stdin.

Evaluation order: deny_rules > allow_rules > mode default > ask. Given the
user's explicit ask that Ducky-driven actions be gated (Ducky is measured
0/10 on bench_ducky.py -- not a reliable actor), PermissionMode.DEFAULT
ships with no pre-populated allow rules: only READ tools (read_file,
list_dir) auto-allow; WRITE/EXEC (write_file, run_shell) ask a human every
time unless an explicit allow rule says otherwise.
"""

from __future__ import annotations

from dataclasses import dataclass

from ducky_agent.permissions.rules import RuleSet
from ducky_agent.permissions.types import PermissionMode, ToolKind


@dataclass
class Decision:
    outcome: str  # "allow" | "deny" | "ask"
    reason: str
    matched_rule: str | None = None


class PermissionGate:
    def __init__(self, mode: PermissionMode = PermissionMode.DEFAULT, rules: RuleSet | None = None):
        self.mode = mode
        self.rules = rules if rules is not None else RuleSet()

    def evaluate(self, tool_name: str, kind: ToolKind, args: dict) -> Decision:
        for rule in self.rules.deny_rules:
            if rule.matches(tool_name, kind, args):
                return Decision("deny", f"denied by rule {rule.id!r}", rule.id)

        for rule in self.rules.allow_rules:
            if rule.matches(tool_name, kind, args):
                return Decision("allow", f"allowed by rule {rule.id!r}", rule.id)

        if self.mode == PermissionMode.YOLO:
            # YOLO only fills the "otherwise ask" gap below -- it never
            # reaches here if a deny rule already matched above.
            return Decision("allow", "YOLO mode: no matching deny rule")

        if kind == ToolKind.READ:
            return Decision("allow", "DEFAULT mode auto-allows READ tools")

        return Decision("ask", "DEFAULT mode asks before WRITE/EXEC tools")
