"""ModelProtocol implementations. ScriptedModel drives the harness through
known cases (Phase 5's verification, and every unit test in this package)
without ever touching torch or a real checkpoint. DuckyModel (wraps the
real ducky.Ducky) is added once the harness mechanics are already verified
correct against ScriptedModel -- see tasks/todo.md's agent-harness phase
for why this ordering matters: bench_ducky.py established the same
scripted-before-real discipline in this repo already.
"""

from __future__ import annotations

from typing import Callable


class ScriptedModel:
    """Either a fixed list of responses (raises loudly once exhausted, so
    a test fails clearly if the loop calls the model more times than
    expected) or a responder(prompt, call_index) -> str callable for cases
    that need to react to the prompt or run indefinitely (e.g. verifying
    max_turns actually halts a model that keeps emitting a new Action
    forever). Exactly one of the two must be given.
    """

    def __init__(
        self,
        responses: list[str] | None = None,
        responder: Callable[[str, int], str] | None = None,
    ):
        if (responses is None) == (responder is None):
            raise ValueError("specify exactly one of responses= or responder=")
        self._responses = responses
        self._responder = responder
        self.call_count = 0
        self.prompts_seen: list[str] = []

    def ask(self, prompt: str) -> str:
        self.prompts_seen.append(prompt)
        idx = self.call_count
        self.call_count += 1
        if self._responder is not None:
            return self._responder(prompt, idx)
        if idx >= len(self._responses):
            raise AssertionError(
                f"ScriptedModel exhausted after {idx} calls (only {len(self._responses)} scripted)"
            )
        return self._responses[idx]


class DuckyModel:
    """Wraps the real ducky.Ducky behind ModelProtocol's single-arg
    ask(prompt) -> str contract. Ducky's own ask() takes several
    generation kwargs (temperature, top_p, repetition_penalty, ...) --
    those are fixed at construction time here (ask_kwargs), not exposed
    per-call, since ModelProtocol is deliberately the narrowest possible
    seam (see model_protocol.py) so ScriptedModel and DuckyModel are
    truly interchangeable to everything above this layer.

    Import of ducky.Ducky is local to __init__, not module-level: importing
    ducky_agent.model_adapter must not force torch/a checkpoint load just
    to reach ScriptedModel, which every test and the parser/gate/tools
    layers depend on being importable with zero torch cost.
    """

    def __init__(self, run_name: str | None = None, **ask_kwargs):
        from ducky import Ducky

        self._ducky = Ducky(run_name=run_name)
        self._ask_kwargs = ask_kwargs

    def ask(self, prompt: str) -> str:
        return self._ducky.ask(prompt, **self._ask_kwargs)
