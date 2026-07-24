"""Matrix-tested permission gate precedence, per the build order's Phase 4
(tasks/todo.md's agent-harness phase): deny > allow > mode default > ask.
Pure logic, no I/O -- the gate itself must never prompt anything."""

from ducky_agent.permissions.gate import PermissionGate
from ducky_agent.permissions.rules import Rule, RuleSet
from ducky_agent.permissions.types import PermissionMode, ToolKind


def test_default_mode_read_auto_allows():
    gate = PermissionGate(mode=PermissionMode.DEFAULT)
    decision = gate.evaluate("read_file", ToolKind.READ, {"path": "x.txt"})
    assert decision.outcome == "allow"


def test_default_mode_write_asks():
    gate = PermissionGate(mode=PermissionMode.DEFAULT)
    decision = gate.evaluate("write_file", ToolKind.WRITE, {"path": "x.txt", "content": "y"})
    assert decision.outcome == "ask"


def test_default_mode_exec_asks():
    gate = PermissionGate(mode=PermissionMode.DEFAULT)
    decision = gate.evaluate("run_shell", ToolKind.EXEC, {"command": "ls"})
    assert decision.outcome == "ask"


def test_default_mode_ships_with_no_prepopulated_allow_rules():
    gate = PermissionGate(mode=PermissionMode.DEFAULT)
    assert gate.rules.allow_rules == []
    assert gate.rules.deny_rules == []


def test_yolo_mode_allows_write_and_exec():
    gate = PermissionGate(mode=PermissionMode.YOLO)
    assert gate.evaluate("write_file", ToolKind.WRITE, {}).outcome == "allow"
    assert gate.evaluate("run_shell", ToolKind.EXEC, {"command": "ls"}).outcome == "allow"


def test_yolo_mode_still_allows_read():
    gate = PermissionGate(mode=PermissionMode.YOLO)
    assert gate.evaluate("read_file", ToolKind.READ, {}).outcome == "allow"


def test_allow_rule_beats_default_mode_ask():
    rules = RuleSet(allow_rules=[Rule(id="allow-writes", kind=ToolKind.WRITE)])
    gate = PermissionGate(mode=PermissionMode.DEFAULT, rules=rules)
    decision = gate.evaluate("write_file", ToolKind.WRITE, {"path": "x.txt", "content": "y"})
    assert decision.outcome == "allow"
    assert decision.matched_rule == "allow-writes"


def test_deny_rule_beats_matching_allow_rule():
    rules = RuleSet(
        deny_rules=[Rule(id="deny-write-file", tool="write_file")],
        allow_rules=[Rule(id="allow-all-writes", kind=ToolKind.WRITE)],
    )
    gate = PermissionGate(mode=PermissionMode.DEFAULT, rules=rules)
    decision = gate.evaluate("write_file", ToolKind.WRITE, {"path": "x.txt", "content": "y"})
    assert decision.outcome == "deny"
    assert decision.matched_rule == "deny-write-file"


def test_deny_rule_beats_yolo_mode():
    rules = RuleSet(deny_rules=[Rule(id="deny-run-shell", tool="run_shell")])
    gate = PermissionGate(mode=PermissionMode.YOLO, rules=rules)
    decision = gate.evaluate("run_shell", ToolKind.EXEC, {"command": "ls"})
    assert decision.outcome == "deny"
    assert decision.matched_rule == "deny-run-shell"


def test_deny_rule_beats_default_mode_read_auto_allow():
    rules = RuleSet(deny_rules=[Rule(id="deny-read-secrets", tool="read_file", arg_contains={"path": "secret"})])
    gate = PermissionGate(mode=PermissionMode.DEFAULT, rules=rules)
    decision = gate.evaluate("read_file", ToolKind.READ, {"path": "/home/user/secrets.env"})
    assert decision.outcome == "deny"


def test_arg_contains_matches_only_matching_calls():
    rules = RuleSet(deny_rules=[Rule(id="deny-rm-rf", tool="run_shell", arg_contains={"command": "rm -rf"})])
    gate = PermissionGate(mode=PermissionMode.YOLO, rules=rules)
    dangerous = gate.evaluate("run_shell", ToolKind.EXEC, {"command": "rm -rf /"})
    safe = gate.evaluate("run_shell", ToolKind.EXEC, {"command": "ls -la"})
    assert dangerous.outcome == "deny"
    assert safe.outcome == "allow"


def test_first_matching_deny_rule_wins_and_stops():
    rules = RuleSet(deny_rules=[
        Rule(id="deny-a", tool="run_shell"),
        Rule(id="deny-b", tool="run_shell"),
    ])
    gate = PermissionGate(mode=PermissionMode.DEFAULT, rules=rules)
    decision = gate.evaluate("run_shell", ToolKind.EXEC, {"command": "ls"})
    assert decision.matched_rule == "deny-a"


def test_tool_specific_rule_does_not_match_other_tools():
    rules = RuleSet(allow_rules=[Rule(id="allow-write-file-only", tool="write_file")])
    gate = PermissionGate(mode=PermissionMode.DEFAULT, rules=rules)
    decision = gate.evaluate("run_shell", ToolKind.EXEC, {"command": "ls"})
    assert decision.outcome == "ask"  # falls through to mode default, not the write_file rule


def test_gate_never_raises_or_blocks_pure_function_call():
    # The gate must be pure logic -- calling evaluate() must never itself
    # attempt any I/O (no input(), no prompt). This is a structural
    # sanity check: a bare call with no stdin/mocking available must
    # simply return.
    gate = PermissionGate(mode=PermissionMode.DEFAULT)
    decision = gate.evaluate("run_shell", ToolKind.EXEC, {"command": "ls"})
    assert decision.outcome == "ask"
