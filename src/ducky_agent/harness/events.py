"""Typed events emitted by loop.run_turn(), one per state-machine step.
Kept as plain dataclasses (not a shared base class with a discriminant
field) so callers pattern-match with isinstance(), matching this repo's own
convention (action_parser.py's ParsedAction/FinalAnswer/ParseError use the
same shape)."""

from __future__ import annotations

from dataclasses import dataclass

from ducky_agent.permissions.types import ToolKind
from ducky_agent.tools.base import ToolResult


@dataclass
class ActionParsed:
    turn: int
    tool: str
    args: dict


@dataclass
class ParseErrorEvent:
    turn: int
    kind: str
    raw_text: str
    detail: str = ""
    retry_number: int = 0
    will_retry: bool = False


@dataclass
class PermissionAsked:
    """Pauses run_turn's generator -- the driving loop must resume via
    generator.send("allow" | "deny")."""

    turn: int
    tool: str
    kind: ToolKind
    args: dict
    reason: str


@dataclass
class PermissionDenied:
    turn: int
    tool: str
    args: dict
    reason: str


@dataclass
class ToolExecuted:
    turn: int
    tool: str
    args: dict
    result: ToolResult


@dataclass
class TurnComplete:
    turn: int
    final_answer: str


@dataclass
class MaxTurnsHit:
    turn: int
