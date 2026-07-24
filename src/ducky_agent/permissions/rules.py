"""Rule matching for the permission gate. A Rule matches on any combination
of tool name / ToolKind / literal arg values -- all fields default to
"matches anything" so a rule can be as broad (deny every run_shell call) or
as narrow (deny only run_shell(command=...) containing a specific string)
as needed.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from ducky_agent.permissions.types import ToolKind


@dataclass
class Rule:
    id: str
    tool: str | None = None  # None matches any tool name
    kind: ToolKind | None = None  # None matches any ToolKind
    # Each (key, substring) pair: the rule matches only if args.get(key) is
    # a string containing substring. Kept to substring matching (not full
    # regex/glob) -- the minimal thing that lets a rule express "deny any
    # run_shell command containing 'rm -rf'" without building a second
    # pattern language.
    arg_contains: dict[str, str] | None = None

    def matches(self, tool_name: str, kind: ToolKind, args: dict) -> bool:
        if self.tool is not None and self.tool != tool_name:
            return False
        if self.kind is not None and self.kind != kind:
            return False
        if self.arg_contains is not None:
            for key, substring in self.arg_contains.items():
                value = args.get(key)
                if not isinstance(value, str) or substring not in value:
                    return False
        return True


@dataclass
class RuleSet:
    deny_rules: list[Rule] = field(default_factory=list)
    allow_rules: list[Rule] = field(default_factory=list)
