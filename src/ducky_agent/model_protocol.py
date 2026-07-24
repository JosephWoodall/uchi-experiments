"""The one seam loop.py depends on -- deliberately the narrowest possible
contract (a single blocking call, string in, string out) so any model can
stand in: the real Ducky() (model_adapter.DuckyModel, Phase 7), or a
ScriptedModel (below) driving the harness through known cases before real,
slow, currently-unreliable inference is ever involved (Phase 5's own
verification point, matching bench_ducky.py's own scripted-first
discipline).
"""

from __future__ import annotations

from typing import Protocol


class ModelProtocol(Protocol):
    def ask(self, prompt: str) -> str: ...
