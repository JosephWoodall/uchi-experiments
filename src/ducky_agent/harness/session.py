"""Synchronous driver for loop.run_turn()'s generator: sends None to
advance, and whenever a PermissionAsked is yielded, calls the supplied
on_ask callback and sends its returned "allow"/"deny" back in to resume.
This is the layer that owns actual human interaction (the gate itself
never prompts, per permissions/gate.py's own docstring) -- here that
interaction is a plain synchronous callback, appropriate for scripts,
tests, and the SDK's default usage; the TUI (a later phase) drives the
same run_turn generator with its own async-native loop instead of this
class, so a real human's modal response doesn't block the Textual event
loop.
"""

from __future__ import annotations

from typing import Callable

from ducky_agent.context.window import PromptWindow
from ducky_agent.harness.events import PermissionAsked
from ducky_agent.loop import run_turn
from ducky_agent.permissions.gate import PermissionGate

OnAsk = Callable[[PermissionAsked], str]


class Session:
    def __init__(self, model, window: PromptWindow, gate: PermissionGate):
        self.model = model
        self.window = window
        self.gate = gate

    def run(
        self,
        task: str,
        on_ask: OnAsk,
        max_turns: int = 8,
        max_parse_retries: int = 2,
        on_event: Callable[[object], None] | None = None,
    ) -> list:
        """Drives one full run_turn generator to completion. Returns the
        full ordered event log (see harness/events.py). ``on_event``, if
        given, is called once per event as it happens -- the one driving
        loop this class owns serves both batch callers (scripts, tests:
        just use the returned list) and live-streaming callers (the TUI:
        update the transcript as each event arrives) without duplicating
        the send/resume logic in a second place.
        """
        events: list = []
        gen = run_turn(self.model, self.window, self.gate, task, max_turns, max_parse_retries)
        send_value = None
        while True:
            try:
                event = gen.send(send_value)
            except StopIteration:
                break
            events.append(event)
            if on_event is not None:
                on_event(event)
            if isinstance(event, PermissionAsked):
                send_value = on_ask(event)
            else:
                send_value = None
        return events
