"""DuckyAgent: the public Python SDK entrypoint -- what
`pip install ducky-agent; from ducky_agent import DuckyAgent` gets you
without the TUI. tui/app.py is itself built on the same Session/loop
underneath.

Carries Ducky.ask()'s own "always returns, never raises" contract up to
this layer too: run() never raises for an expected agent-loop condition
(a parse failure, a permission denial, hitting max_turns) -- all of those
are represented in the returned AgentResult's event log, not exceptions.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

from ducky_agent.context.window import PromptWindow
from ducky_agent.harness.events import MaxTurnsHit, PermissionAsked, TurnComplete
from ducky_agent.harness.session import Session
from ducky_agent.permissions.gate import PermissionGate
from ducky_agent.permissions.types import PermissionMode
from ducky_agent.tools.registry import TOOL_REGISTRY


@dataclass
class AgentResult:
    final_answer: str | None  # None if max_turns was hit before ever completing
    events: list = field(default_factory=list)
    hit_max_turns: bool = False


class DuckyAgent:
    def __init__(
        self,
        model=None,
        permission_mode: PermissionMode = PermissionMode.DEFAULT,
        max_turns: int = 8,
        max_parse_retries: int = 2,
    ):
        """model defaults to a real DuckyModel() wrapping ducky.Ducky --
        injectable with model_adapter.ScriptedModel for scripting/tests
        with zero torch/checkpoint cost. The tool set is currently fixed
        (tools/registry.py's 4-tool minimal set, see the plan); no
        pluggable-tools parameter is exposed here since nothing beneath
        this layer is actually parameterized by it yet -- adding an
        unused kwarg would be dishonest surface area.
        """
        if model is None:
            from ducky_agent.model_adapter import DuckyModel

            model = DuckyModel()
        self.model = model
        self.gate = PermissionGate(mode=permission_mode)
        self.window = PromptWindow(TOOL_REGISTRY)
        self.session = Session(model=self.model, window=self.window, gate=self.gate)
        self.max_turns = max_turns
        self.max_parse_retries = max_parse_retries

    def run(
        self,
        task: str,
        on_ask: Callable[[PermissionAsked], str] | None = None,
    ) -> AgentResult:
        """on_ask defaults to always "deny" -- the safe choice for a
        headless/programmatic call with no human present, consistent with
        this project's own ask-by-default posture (see permissions/gate.py).
        A caller that wants auto-approval passes on_ask=lambda e: "allow",
        or constructs with permission_mode=PermissionMode.YOLO instead
        (which mostly avoids asking at all, only still honoring explicit
        deny rules).
        """
        resolved_on_ask = on_ask or (lambda event: "deny")
        events = self.session.run(
            task,
            on_ask=resolved_on_ask,
            max_turns=self.max_turns,
            max_parse_retries=self.max_parse_retries,
        )

        final_answer = None
        hit_max_turns = False
        for event in events:
            if isinstance(event, TurnComplete):
                final_answer = event.final_answer
            elif isinstance(event, MaxTurnsHit):
                hit_max_turns = True

        return AgentResult(final_answer=final_answer, events=events, hit_max_turns=hit_max_turns)
