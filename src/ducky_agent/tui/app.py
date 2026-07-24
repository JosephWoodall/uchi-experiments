"""Ducky agent TUI: single screen, persistent input, permission modal.

Normal mode: Input submit kicks off a threaded worker (must not block the
event loop -- generation is real wall-clock time on CPU/GPU, and even a
ScriptedModel-driven run should not freeze the UI). Awaiting mode: the
worker's on_ask bridge shows PermissionModal on the main thread and blocks
(in the worker thread only, never the UI thread) until it resolves, then
resumes the paused run_turn generator via harness.session.Session's own
send/resume driver -- reused as-is, not duplicated, via its on_event
callback for live per-event transcript updates.

--fake-model wires this same app to ScriptedModel (model_adapter.py) for
human-observed verification of the TUI mechanics themselves, before ever
pointing it at slow, currently-unreliable real Ducky inference.
"""

from __future__ import annotations

import argparse
import threading

from textual import work
from textual.app import App, ComposeResult
from textual.widgets import Footer, Header, Input

from ducky_agent.context.window import PromptWindow
from ducky_agent.harness.events import PermissionAsked
from ducky_agent.harness.session import Session
from ducky_agent.model_adapter import ScriptedModel
from ducky_agent.permissions.gate import PermissionGate
from ducky_agent.permissions.rules import Rule
from ducky_agent.permissions.types import PermissionMode
from ducky_agent.tools.registry import TOOL_REGISTRY
from ducky_agent.tui.widgets import DismissResult, PermissionModal, format_event, make_transcript


class DuckyAgentApp(App):
    CSS = """
    #transcript { border: round $primary; height: 1fr; }
    """
    BINDINGS = [("ctrl+c", "quit", "Quit")]

    def __init__(
        self,
        model,
        permission_mode: PermissionMode = PermissionMode.DEFAULT,
        max_turns: int = 8,
    ):
        super().__init__()
        self.model = model
        self.gate = PermissionGate(mode=permission_mode)
        self.session = Session(model=self.model, window=PromptWindow(TOOL_REGISTRY), gate=self.gate)
        self.max_turns = max_turns
        self.turn_count = 0
        self._busy = False

    def compose(self) -> ComposeResult:
        yield Header(show_clock=True)
        yield make_transcript()
        yield Input(placeholder="Describe a task for Ducky...", id="task-input")
        yield Footer()

    def on_mount(self) -> None:
        self._refresh_subtitle()

    def _refresh_subtitle(self) -> None:
        self.sub_title = f"mode={self.gate.mode.value} turn={self.turn_count}"

    def on_input_submitted(self, event: Input.Submitted) -> None:
        if self._busy:
            return
        task = event.value.strip()
        if not task:
            return
        self.query_one("#task-input", Input).value = ""
        self.query_one("#transcript").write(f"[bold]> {task}[/]")
        self._busy = True
        self.run_agent_turn(task)

    @work(thread=True, exclusive=True)
    def run_agent_turn(self, task: str) -> None:
        def on_ask(event: PermissionAsked) -> str:
            decision_ready = threading.Event()
            outcome_box: dict = {}

            def show_modal() -> None:
                def handle_result(result: DismissResult) -> None:
                    outcome, scope = result
                    if outcome == "allow" and scope == "always":
                        self.gate.rules.allow_rules.append(
                            Rule(id=f"tui-always-{event.tool}", tool=event.tool)
                        )
                    outcome_box["outcome"] = outcome
                    decision_ready.set()

                self.push_screen(PermissionModal(event), handle_result)

            self.call_from_thread(show_modal)
            decision_ready.wait()
            return outcome_box["outcome"]

        def on_event(event: object) -> None:
            self.call_from_thread(self._append_event, event)

        self.session.run(task, on_ask=on_ask, max_turns=self.max_turns, on_event=on_event)
        self.call_from_thread(self._mark_idle)

    def _append_event(self, event: object) -> None:
        turn = getattr(event, "turn", None)
        if turn is not None:
            self.turn_count = turn
        self.query_one("#transcript").write(format_event(event))
        self._refresh_subtitle()

    def _mark_idle(self) -> None:
        self._busy = False


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="ducky-agent")
    p.add_argument(
        "--fake-model",
        action="store_true",
        help="Drive the TUI with a ScriptedModel instead of real Ducky -- for human-observed "
        "verification of the TUI mechanics themselves.",
    )
    p.add_argument(
        "--yolo",
        action="store_true",
        help="Start in PermissionMode.YOLO (auto-allow everything except explicit deny rules) "
        "instead of the DEFAULT ask-before-write/exec mode.",
    )
    p.add_argument("--max-turns", type=int, default=8)
    return p


def _demo_scripted_model() -> ScriptedModel:
    """A small fixed script exercising every mechanic (a read tool call, a
    write requiring permission, a final answer) -- enough to verify the
    TUI by eye without a real checkpoint."""
    return ScriptedModel(
        responses=[
            'Thought: let me look around.\nAction: list_dir(path=".")',
            'Thought: now write a note.\n'
            'Action: write_file(path="ducky_agent_demo.txt", content="hello from the fake model")',
            "Done -- listed the directory and wrote a demo file.",
        ]
    )


def main() -> None:
    args = build_parser().parse_args()
    mode = PermissionMode.YOLO if args.yolo else PermissionMode.DEFAULT

    if args.fake_model:
        model = _demo_scripted_model()
    else:
        # Local import: only touches torch/Ducky's checkpoint loading if a
        # real run is actually requested.
        from ducky_agent.model_adapter import DuckyModel

        model = DuckyModel()

    app = DuckyAgentApp(model=model, permission_mode=mode, max_turns=args.max_turns)
    app.run()


if __name__ == "__main__":
    main()
