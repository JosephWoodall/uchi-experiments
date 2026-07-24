"""bench_ducky.py-style scripted end-to-end check for the agent harness
(ducky_agent), run against ScriptedModel -- never real Ducky, never real
GPU/CPU inference cost. Same discipline this repo already established for
run_sandboxed (grounding.py): verify the mechanism against known-good,
known-bad, and known-infinite scripted cases BEFORE trusting any number
from the real (currently 0/10-on-bench_ducky.py) model. Prints PASS/FAIL
per scenario and exits non-zero on any failure.
"""
import sys
import tempfile
from pathlib import Path

from ducky_agent.context.window import PromptWindow
from ducky_agent.harness.events import (
    ActionParsed,
    MaxTurnsHit,
    ParseErrorEvent,
    PermissionDenied,
    ToolExecuted,
    TurnComplete,
)
from ducky_agent.harness.session import Session
from ducky_agent.model_adapter import ScriptedModel
from ducky_agent.permissions.gate import PermissionGate
from ducky_agent.permissions.types import PermissionMode
from ducky_agent.tools.registry import TOOL_REGISTRY


def check_canned_correct_action_then_final_answer() -> tuple[bool, str]:
    with tempfile.TemporaryDirectory() as tmp:
        (Path(tmp) / "a.txt").write_text("hello")
        model = ScriptedModel(
            responses=[
                f'Thought: list the dir\nAction: list_dir(path="{tmp}")',
                "The directory contains one file, a.txt.",
            ]
        )
        session = Session(model=model, window=PromptWindow(TOOL_REGISTRY), gate=PermissionGate(PermissionMode.YOLO))
        events = session.run("list the files", on_ask=lambda e: "allow", max_turns=4)

    executed = [e for e in events if isinstance(e, ToolExecuted)]
    complete = [e for e in events if isinstance(e, TurnComplete)]
    if len(executed) != 1 or not executed[0].result.ok:
        return False, f"expected 1 successful ToolExecuted, got {executed}"
    if len(complete) != 1 or "a.txt" not in complete[0].final_answer:
        return False, f"expected a clean TurnComplete mentioning a.txt, got {complete}"
    if "a.txt" not in model.prompts_seen[1]:
        return False, "tool observation was not fed back into the next prompt"
    return True, f"1 tool executed, observation fed back, finished at turn {complete[0].turn}"


def check_malformed_action_retries_then_succeeds() -> tuple[bool, str]:
    with tempfile.TemporaryDirectory() as tmp:
        model = ScriptedModel(
            responses=[
                "Action: list_dir(",  # malformed syntax, retry 1
                f'Action: list_dir(path="{tmp}")',  # valid, executes
                "Nothing interesting here.",  # final answer, turn 2
            ]
        )
        session = Session(model=model, window=PromptWindow(TOOL_REGISTRY), gate=PermissionGate(PermissionMode.YOLO))
        events = session.run("look around", on_ask=lambda e: "allow", max_turns=4)

    parse_errors = [e for e in events if isinstance(e, ParseErrorEvent)]
    executed = [e for e in events if isinstance(e, ToolExecuted)]
    complete = [e for e in events if isinstance(e, TurnComplete)]
    if len(parse_errors) != 1 or not parse_errors[0].will_retry:
        return False, f"expected exactly 1 retryable ParseErrorEvent, got {parse_errors}"
    if len(executed) != 1:
        return False, f"expected the retry to succeed and execute once, got {executed}"
    if len(complete) != 1:
        return False, f"expected a clean finish, got {complete}"
    return True, "malformed action didn't crash the loop, retried successfully, then finished"


def check_exhausted_retries_falls_back_to_raw_text() -> tuple[bool, str]:
    model = ScriptedModel(responder=lambda prompt, i: "Action: totally(broken(")
    session = Session(model=model, window=PromptWindow(TOOL_REGISTRY), gate=PermissionGate(PermissionMode.YOLO))
    events = session.run("do something", on_ask=lambda e: "allow", max_turns=4, max_parse_retries=2)

    parse_errors = [e for e in events if isinstance(e, ParseErrorEvent)]
    complete = [e for e in events if isinstance(e, TurnComplete)]
    if len(parse_errors) != 3:  # 1 initial + 2 retries, all in turn 1
        return False, f"expected 3 ParseErrorEvents (1 initial + 2 retries), got {len(parse_errors)}"
    if parse_errors[-1].will_retry:
        return False, "last ParseErrorEvent should have will_retry=False"
    if len(complete) != 1 or model.call_count != 3:
        return False, f"expected graceful fallback to a raw-text FinalAnswer after 3 calls, got {complete}, calls={model.call_count}"
    return True, "exhausted retries fell back to a raw-text answer instead of crashing or hanging"


def check_infinite_action_model_halted_by_max_turns() -> tuple[bool, str]:
    with tempfile.TemporaryDirectory() as tmp:
        model = ScriptedModel(responder=lambda prompt, i: f'Action: list_dir(path="{tmp}")')
        session = Session(model=model, window=PromptWindow(TOOL_REGISTRY), gate=PermissionGate(PermissionMode.YOLO))
        events = session.run("loop forever", on_ask=lambda e: "allow", max_turns=3)

    executed = [e for e in events if isinstance(e, ToolExecuted)]
    max_turns_hit = [e for e in events if isinstance(e, MaxTurnsHit)]
    if len(executed) != 3:
        return False, f"expected exactly 3 tool executions (one per turn), got {len(executed)}"
    if len(max_turns_hit) != 1 or max_turns_hit[0].turn != 3:
        return False, f"expected MaxTurnsHit(turn=3), got {max_turns_hit}"
    return True, "a model that never stops emitting Actions was halted cleanly at max_turns"


def check_permission_deny_produces_zero_mutation() -> tuple[bool, str]:
    with tempfile.TemporaryDirectory() as tmp:
        target = str(Path(tmp) / "out.txt")
        model = ScriptedModel(
            responses=[
                f'Action: write_file(path="{target}", content="should not exist")',
                "Done.",
            ]
        )
        session = Session(model=model, window=PromptWindow(TOOL_REGISTRY), gate=PermissionGate(PermissionMode.DEFAULT))
        events = session.run("write a file", on_ask=lambda e: "deny", max_turns=4)

        denied = [e for e in events if isinstance(e, PermissionDenied)]
        executed = [e for e in events if isinstance(e, ToolExecuted)]
        file_exists = Path(target).exists()

    if len(denied) != 1:
        return False, f"expected exactly 1 PermissionDenied, got {denied}"
    if executed:
        return False, f"expected zero ToolExecuted after a deny, got {executed}"
    if file_exists:
        return False, "DENY produced a real filesystem mutation -- this is the critical safety check"
    return True, "deny produced zero filesystem mutation, verified on real disk"


def check_permission_allow_actually_executes() -> tuple[bool, str]:
    with tempfile.TemporaryDirectory() as tmp:
        target = str(Path(tmp) / "out.txt")
        model = ScriptedModel(
            responses=[
                f'Action: write_file(path="{target}", content="written for real")',
                "Done.",
            ]
        )
        session = Session(model=model, window=PromptWindow(TOOL_REGISTRY), gate=PermissionGate(PermissionMode.DEFAULT))
        events = session.run("write a file", on_ask=lambda e: "allow", max_turns=4)

        executed = [e for e in events if isinstance(e, ToolExecuted)]
        on_disk = Path(target).read_text() if Path(target).exists() else None

    if len(executed) != 1 or not executed[0].result.ok:
        return False, f"expected exactly 1 successful ToolExecuted, got {executed}"
    if on_disk != "written for real":
        return False, f"expected the real file content on disk, got {on_disk!r}"
    return True, "allow actually wrote the real file, verified on real disk"


CHECKS = [
    check_canned_correct_action_then_final_answer,
    check_malformed_action_retries_then_succeeds,
    check_exhausted_retries_falls_back_to_raw_text,
    check_infinite_action_model_halted_by_max_turns,
    check_permission_deny_produces_zero_mutation,
    check_permission_allow_actually_executes,
]


def main() -> int:
    n_pass = 0
    for check in CHECKS:
        try:
            passed, detail = check()
        except Exception as e:  # a harness bug, not an expected failure -- surface it, don't hide it
            passed, detail = False, f"raised {type(e).__name__}: {e}"
        status = "PASS" if passed else "FAIL"
        print(f"[{status}] {check.__name__}: {detail}")
        n_pass += passed

    print(f"\n{n_pass}/{len(CHECKS)} checks passed")
    return 0 if n_pass == len(CHECKS) else 1


if __name__ == "__main__":
    sys.exit(main())
