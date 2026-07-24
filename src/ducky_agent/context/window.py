"""Assembles the prompt for each loop iteration: a pinned Thought/Action
syntax preamble + the growing transcript of this agent run + the current
task/observation.

Reuses session_history.SessionHistory directly for transcript compaction
(extractive, verbatim, bounded -- never an abstractive summary sitting in
the model's own context) rather than inventing a second compaction
mechanism for what is, mechanically, the same problem SessionHistory
already solves: a growing log of prompt/response pairs that needs to stay
bounded. Each loop iteration's real model.ask() call is recorded as one
SessionHistory entry.
"""

from __future__ import annotations

from session_history import SessionHistory

SYNTAX_PREAMBLE = """You can call tools using this exact format:
Thought: <your reasoning>
Action: tool_name(key="value")

If you are done and have a final answer, respond with plain text and no Action line.

Available tools:
{tool_descriptions}

Example:
Thought: I need to see what files exist.
Action: list_dir(path=".")"""


def build_tool_descriptions(tool_specs: dict) -> str:
    return "\n".join(f"- {name}: {spec.description}" for name, spec in tool_specs.items())


class PromptWindow:
    def __init__(self, tool_specs: dict, max_log_chars: int = 2000):
        self.history = SessionHistory()
        self.preamble = SYNTAX_PREAMBLE.format(
            tool_descriptions=build_tool_descriptions(tool_specs)
        )
        self.max_log_chars = max_log_chars

    def build_prompt(self, task: str, observation: str | None) -> str:
        parts = [self.preamble, f"Task: {task}"]
        ctx = self.history.context_string()
        if ctx:
            parts.append(ctx)
        if observation is not None:
            parts.append(f"Observation: {observation}")
        return "\n\n".join(parts)

    def record(self, prompt: str, response: str) -> None:
        self.history.record(prompt, response)
        self.history.maybe_compact(max_chars=self.max_log_chars)
