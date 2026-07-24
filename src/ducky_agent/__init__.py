"""Ducky's agent harness: a public Python SDK plus a terminal UI, built on
top of the raw ``Ducky`` next-token predictor (../ducky.py).

Ducky has zero instruction-tuning and zero native tool-calling -- this
package's ``action_parser`` extracts tool calls from free-text generation
via a fixed Thought/Action convention instead of relying on a structured
API a provider's model would emit natively. See tasks/ducky.md and
tasks/core_principle.md for why (this is the agent-capability half of the
repo's "Ducky as unified engine" objective), and
tasks/todo.md's agent-harness phase for the honestly-measured real numbers.
"""

__version__ = "0.1.0"

from ducky_agent.sdk import AgentResult, DuckyAgent

__all__ = ["DuckyAgent", "AgentResult"]
