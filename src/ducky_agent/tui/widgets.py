"""Transcript rendering + the permission modal. Kept separate from app.py
so the modal's dismiss-result contract (below) can be unit-tested/driven
by Textual's headless Pilot without needing the full App wired up.
"""

from __future__ import annotations

from textual.app import ComposeResult
from textual.containers import Vertical
from textual.screen import ModalScreen
from textual.widgets import Button, Label, RichLog, Static

from ducky_agent.harness.events import (
    ActionParsed,
    MaxTurnsHit,
    ParseErrorEvent,
    PermissionAsked,
    PermissionDenied,
    ToolExecuted,
    TurnComplete,
)

# Dismiss-result contract: ("allow" | "deny", "once" | "always"). The
# "always" half is only meaningful for an allow -- the worker adds a
# session-scoped allow rule to the live gate before resuming, matching the
# reference harness's Allow-once/Deny/Always-for-this-rule modal shape.
DismissResult = tuple[str, str]


class PermissionModal(ModalScreen[DismissResult]):
    """Shows the exact pending tool call and blocks the (worker-thread)
    caller until the human resolves it -- see tui/app.py's on_ask bridge
    for how a synchronous worker-thread wait couples to this async modal.
    """

    DEFAULT_CSS = """
    PermissionModal {
        align: center middle;
    }
    #permission-box {
        width: 60;
        height: auto;
        border: thick $warning;
        background: $surface;
        padding: 1 2;
    }
    #permission-buttons {
        height: auto;
        align: center middle;
        margin-top: 1;
    }
    #permission-buttons Button {
        margin: 0 1;
    }
    """

    def __init__(self, event: PermissionAsked):
        super().__init__()
        self._event = event

    def compose(self) -> ComposeResult:
        e = self._event
        args_str = ", ".join(f"{k}={v!r}" for k, v in e.args.items())
        with Vertical(id="permission-box"):
            yield Label(f"Permission requested (turn {e.turn})", classes="title")
            yield Static(f"{e.tool}({args_str})")
            yield Static(e.reason, classes="reason")
            with Vertical(id="permission-buttons"):
                yield Button("Allow once", id="allow-once", variant="success")
                yield Button("Always allow this tool", id="allow-always", variant="warning")
                yield Button("Deny", id="deny", variant="error")

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "allow-once":
            self.dismiss(("allow", "once"))
        elif event.button.id == "allow-always":
            self.dismiss(("allow", "always"))
        else:
            self.dismiss(("deny", "once"))


def format_event(event: object) -> str:
    """Renders one harness event as a transcript line. Rich markup ([dim],
    [bold], etc.) -- app.py's RichLog is constructed with markup=True."""
    if isinstance(event, ActionParsed):
        args_str = ", ".join(f"{k}={v!r}" for k, v in event.args.items())
        return f"[bold cyan]Action[/] (turn {event.turn}): {event.tool}({args_str})"
    if isinstance(event, ParseErrorEvent):
        retry_note = "retrying" if event.will_retry else "giving up, falling back to raw text"
        return f"[yellow]Parse error[/] (turn {event.turn}, {event.kind}): {retry_note}"
    if isinstance(event, PermissionAsked):
        return f"[bold magenta]Permission requested[/] (turn {event.turn}): {event.tool}"
    if isinstance(event, PermissionDenied):
        return f"[red]Denied[/] (turn {event.turn}): {event.tool} -- {event.reason}"
    if isinstance(event, ToolExecuted):
        status = "[green]ok[/]" if event.result.ok else "[red]failed[/]"
        return f"[bold]Observation[/] (turn {event.turn}, {status}): {event.result.output}"
    if isinstance(event, TurnComplete):
        return f"[bold green]Final answer[/] (turn {event.turn}): {event.final_answer}"
    if isinstance(event, MaxTurnsHit):
        return f"[bold red]Max turns reached[/] (turn {event.turn}) -- stopping."
    return str(event)


def make_transcript() -> RichLog:
    return RichLog(markup=True, wrap=True, highlight=False, id="transcript")
