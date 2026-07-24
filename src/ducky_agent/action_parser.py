"""Extracts a tool call from Ducky's raw free-text generation.

Ducky has zero instruction-tuning and zero native function-calling -- there
is no structured-output API to lean on (contrast the reference harness this
package is modeled after, which relies entirely on the provider LLM's own
tool-call emission). The only mechanism available is a fixed, literal
Thought/Action example transcript in the prompt preamble that Ducky
pattern-matches via raw text continuation:

    Thought: <free text, logged only, never parsed>
    Action: tool_name(key="value", key2="value2")

Given the same base capability that scores 0/10 on much simpler docstring
completion (bench_ducky.py), producing no parseable Action at all is the
EXPECTED default outcome here, not an edge case -- callers must treat
FinalAnswer as the common path, not a fallback.

Only the first ``^Action:`` line is honored (re.MULTILINE, single-line
capture): Ducky's known repetitive-degeneration failure mode (tasks/ducky.md
Phase L) makes later lines unreliable once a generation starts looping.

The call itself is validated with ast.parse(mode="eval") restricted to a
single Call(Name, all-keyword Constant args) -- the same restricted-AST
idiom this repo already uses for untrusted-model-output parsing
(grounding.py's _SAFE_BUILTINS / arithmetic_grounded), not a hand-rolled
regex or arbitrary eval(). Positional args are rejected on purpose: tool
schemas are named, and keyword-only calls are unambiguous to validate.
"""

from __future__ import annotations

import ast
import re
from dataclasses import dataclass, field

_ACTION_LINE_RE = re.compile(r"^Action:\s*(.+)$", re.MULTILINE)

# What a Constant's runtime value is allowed to be -- excludes bytes/complex/
# tuples-of-constants etc., which ast.Constant can technically hold but a
# tool-call argument never legitimately needs.
_ALLOWED_CONSTANT_TYPES = (str, int, float, bool, type(None))


@dataclass
class ParsedAction:
    """A successfully parsed, validated tool call."""

    tool: str
    args: dict = field(default_factory=dict)
    raw_text: str = ""


@dataclass
class FinalAnswer:
    """No ``Action:`` line found -- the model's full generation is treated
    as its answer. The expected common outcome, not a failure mode."""

    text: str


@dataclass
class ParseError:
    """An ``Action:`` line was found but couldn't be turned into a valid,
    known tool call. ``kind`` is one of: "syntax", "not_a_call",
    "invalid_func", "positional_args", "starred_args", "invalid_arg_type",
    "unknown_tool"."""

    kind: str
    raw_text: str
    detail: str = ""


def parse_action(text: str, known_tools: frozenset[str]) -> ParsedAction | FinalAnswer | ParseError:
    """Parse one generation's worth of text. ``known_tools`` is the live
    tool registry's name set -- passed in rather than imported, so this
    module has no dependency on tools/registry.py and stays testable in
    complete isolation (Phase 2 of the build order runs before Phase 3's
    tools exist at all).
    """
    match = _ACTION_LINE_RE.search(text)
    if match is None:
        return FinalAnswer(text=text.strip())

    call_str = match.group(1).strip()

    try:
        tree = ast.parse(call_str, mode="eval")
    except SyntaxError as e:
        return ParseError(kind="syntax", raw_text=call_str, detail=str(e))

    call = tree.body
    if not isinstance(call, ast.Call):
        return ParseError(kind="not_a_call", raw_text=call_str)

    if not isinstance(call.func, ast.Name):
        return ParseError(kind="invalid_func", raw_text=call_str)
    tool_name = call.func.id

    if call.args:
        return ParseError(kind="positional_args", raw_text=call_str)

    args: dict = {}
    for kw in call.keywords:
        if kw.arg is None:  # **kwargs expansion, e.g. f(**d)
            return ParseError(kind="starred_args", raw_text=call_str)
        if not isinstance(kw.value, ast.Constant) or not isinstance(
            kw.value.value, _ALLOWED_CONSTANT_TYPES
        ):
            return ParseError(kind="invalid_arg_type", raw_text=call_str, detail=kw.arg)
        args[kw.arg] = kw.value.value

    if tool_name not in known_tools:
        return ParseError(kind="unknown_tool", raw_text=call_str, detail=tool_name)

    return ParsedAction(tool=tool_name, args=args, raw_text=call_str)
