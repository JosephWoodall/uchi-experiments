"""SessionHistory: extractive, bounded cross-call memory for one Ducky()
instance's lifetime -- modeled on uchi's goal_state.py, adapted from "tool-
call log compaction across one agentic task" to "ask() call log compaction
across one Ducky() session." RAM-only by design, same as session_memory.py's
per-generation trie (see tasks/ducky.md for why persistence to disk was
explicitly decided against) -- this is cross-CALL memory within one running
process, not cross-PROCESS memory.

Compaction is extractive, not abstractive, for the same reason
goal_state.py gives: a paraphrase engine is a confabulation risk sitting
right in the middle of what gets fed back to the model as context. Every
note traces back to an actual prior prompt/completion, verbatim (bounded to
excerpt_chars), never a summary invented on the fly.
"""
from dataclasses import dataclass, field

DEFAULT_MAX_LOG_CHARS = 2000
DEFAULT_EXCERPT_CHARS = 150


@dataclass
class SessionHistory:
    notes: list = field(default_factory=list)
    raw_log: list = field(default_factory=list)  # [{"prompt": str, "response": str}, ...]
    compacted_count: int = 0

    def record(self, prompt: str, response: str) -> None:
        self.raw_log.append({"prompt": prompt, "response": response})

    def raw_log_chars(self) -> int:
        return sum(len(e["prompt"]) + len(e["response"]) for e in self.raw_log)

    def maybe_compact(self, max_chars: int = DEFAULT_MAX_LOG_CHARS,
                       excerpt_chars: int = DEFAULT_EXCERPT_CHARS) -> bool:
        """If the raw call log exceeds max_chars, compact it into notes
        (verbatim excerpts, never a paraphrase) and drop the raw entries.
        Returns whether it ran, same contract as goal_state.py's.
        """
        if self.raw_log_chars() <= max_chars:
            return False
        for entry in self.raw_log:
            excerpt = entry["response"][:excerpt_chars]
            if len(entry["response"]) > excerpt_chars:
                excerpt += "..."
            self.notes.append(f"asked {entry['prompt'][:40]!r} -> {excerpt!r}")
            self.compacted_count += 1
        self.raw_log.clear()
        return True

    def context_string(self) -> str:
        """Renders accumulated history as text to fold back into the next
        prompt -- empty string if there's nothing yet, so callers can
        always safely prepend this without a None-check.
        """
        if not self.notes and not self.raw_log:
            return ""
        parts = ["Prior in this session:"]
        parts.extend(f"- {n}" for n in self.notes)
        for entry in self.raw_log:
            excerpt = entry["response"][:DEFAULT_EXCERPT_CHARS]
            parts.append(f"- asked {entry['prompt'][:40]!r} -> {excerpt!r}")
        return "\n".join(parts)
