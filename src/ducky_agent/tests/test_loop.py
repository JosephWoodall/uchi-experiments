"""Unit-level tests for loop.run_turn()'s generator contract directly (not
through harness/session.py's driver) -- complements
src/verify_agent_harness.py's flat, bench_ducky.py-style scripted
end-to-end checks with pytest-native coverage of the send/yield contract
itself."""

from ducky_agent.context.window import PromptWindow
from ducky_agent.harness.events import (
    ActionParsed,
    PermissionAsked,
    ToolExecuted,
    TurnComplete,
)
from ducky_agent.loop import run_turn
from ducky_agent.model_adapter import ScriptedModel
from ducky_agent.permissions.gate import PermissionGate
from ducky_agent.permissions.types import PermissionMode
from ducky_agent.tools.registry import TOOL_REGISTRY


def _drive(gen, on_ask=lambda e: "allow"):
    """Same send-loop as harness/session.py's Session.run, kept local so
    this test doesn't depend on Session working correctly to test loop.py
    in isolation."""
    events = []
    send_value = None
    while True:
        try:
            event = gen.send(send_value)
        except StopIteration:
            break
        events.append(event)
        send_value = on_ask(event) if isinstance(event, PermissionAsked) else None
    return events


def test_final_answer_with_no_action_completes_immediately(tmp_path):
    model = ScriptedModel(responses=["Just a plain answer, no Action line."])
    window = PromptWindow(TOOL_REGISTRY)
    gate = PermissionGate(PermissionMode.YOLO)
    events = _drive(run_turn(model, window, gate, "task", max_turns=4))
    assert len(events) == 1
    assert isinstance(events[0], TurnComplete)
    assert events[0].turn == 1


def test_permission_asked_pauses_and_resumes_on_send(tmp_path):
    target = str(tmp_path / "out.txt")
    model = ScriptedModel(
        responses=[f'Action: write_file(path="{target}", content="x")', "done"]
    )
    window = PromptWindow(TOOL_REGISTRY)
    gate = PermissionGate(PermissionMode.DEFAULT)  # write asks
    gen = run_turn(model, window, gate, "task", max_turns=4)

    first = gen.send(None)
    assert isinstance(first, ActionParsed)
    second = gen.send(None)
    assert isinstance(second, PermissionAsked)
    assert second.tool == "write_file"

    third = gen.send("allow")
    assert isinstance(third, ToolExecuted)
    assert third.result.ok is True
    assert (tmp_path / "out.txt").read_text() == "x"


def test_default_mode_read_tool_never_asks(tmp_path):
    model = ScriptedModel(responses=[f'Action: list_dir(path="{tmp_path}")', "done"])
    window = PromptWindow(TOOL_REGISTRY)
    gate = PermissionGate(PermissionMode.DEFAULT)  # read auto-allows
    events = _drive(run_turn(model, window, gate, "task", max_turns=4))
    assert not any(isinstance(e, PermissionAsked) for e in events)
    assert any(isinstance(e, ToolExecuted) and e.result.ok for e in events)


def test_observation_from_tool_flows_into_next_prompt(tmp_path):
    (tmp_path / "marker_file.txt").write_text("x")
    model = ScriptedModel(
        responses=[f'Action: list_dir(path="{tmp_path}")', "saw the marker file"]
    )
    window = PromptWindow(TOOL_REGISTRY)
    gate = PermissionGate(PermissionMode.YOLO)
    _drive(run_turn(model, window, gate, "task", max_turns=4))
    assert "marker_file.txt" in model.prompts_seen[1]
