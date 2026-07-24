"""The per-turn agent loop: BuildPrompt -> Generate -> Parse -> Gate ->
Execute/Ask -> Observation, repeated until a FinalAnswer or max_turns.

A generator, not a plain function: PermissionAsked is yielded and the
resuming decision is sent back in via generator.send("allow" | "deny") --
the same yield-before-a-pause-point shape the reference harness's own turn
handler uses (its agent.iter() pauses a turn at DeferredToolRequests and
resumes with DeferredToolResults), adapted to a synchronous generator here
since Ducky's ask() is itself a blocking call and this layer has no async
work of its own. harness/session.py drives the generator synchronously;
the TUI (a later phase) drives the same generator inside an async worker
so real generation time never blocks the event loop.

Given the same base capability that scores 0/10 on much simpler docstring
completion (bench_ducky.py), a ParseError on nearly every call is the
expected common outcome here, not a bug to chase -- this loop's job is to
handle that gracefully (retry with the real error spliced in, then give up
to a plain-text answer), never to crash or hang.
"""

from __future__ import annotations

from ducky_agent.action_parser import FinalAnswer, ParseError, ParsedAction, parse_action
from ducky_agent.context.window import PromptWindow
from ducky_agent.harness.events import (
    ActionParsed,
    MaxTurnsHit,
    ParseErrorEvent,
    PermissionAsked,
    PermissionDenied,
    ToolExecuted,
    TurnComplete,
)
from ducky_agent.permissions.gate import PermissionGate
from ducky_agent.tools.registry import TOOL_REGISTRY, execute_tool, known_tool_names


def run_turn(
    model,
    window: PromptWindow,
    gate: PermissionGate,
    task: str,
    max_turns: int = 8,
    max_parse_retries: int = 2,
):
    observation = None
    known_tools = known_tool_names()

    for turn in range(1, max_turns + 1):
        prompt = window.build_prompt(task, observation)
        retry_count = 0

        while True:
            response = model.ask(prompt)
            window.record(prompt, response)
            parsed = parse_action(response, known_tools)

            if not isinstance(parsed, ParseError):
                break

            will_retry = retry_count < max_parse_retries
            yield ParseErrorEvent(
                turn=turn,
                kind=parsed.kind,
                raw_text=parsed.raw_text,
                detail=parsed.detail,
                retry_number=retry_count,
                will_retry=will_retry,
            )
            if not will_retry:
                # Give up gracefully -- treat the raw text as the answer
                # rather than crashing or looping forever, matching
                # Ducky.ask()'s own "always returns" contract at this
                # layer too.
                parsed = FinalAnswer(text=response.strip())
                break

            retry_count += 1
            prompt = window.build_prompt(
                task,
                f"Your last Action could not be parsed ({parsed.kind}): "
                f"{parsed.raw_text!r}. Use the exact syntax: "
                'Action: tool_name(key="value")',
            )

        if isinstance(parsed, FinalAnswer):
            yield TurnComplete(turn=turn, final_answer=parsed.text)
            return

        assert isinstance(parsed, ParsedAction)
        spec = TOOL_REGISTRY[parsed.tool]
        yield ActionParsed(turn=turn, tool=parsed.tool, args=parsed.args)

        decision = gate.evaluate(parsed.tool, spec.kind, parsed.args)
        if decision.outcome == "ask":
            outcome = yield PermissionAsked(
                turn=turn,
                tool=parsed.tool,
                kind=spec.kind,
                args=parsed.args,
                reason=decision.reason,
            )
        else:
            outcome = decision.outcome

        if outcome == "deny":
            yield PermissionDenied(
                turn=turn, tool=parsed.tool, args=parsed.args, reason=decision.reason
            )
            observation = f"Permission denied for {parsed.tool}({parsed.args})."
            continue

        result = execute_tool(parsed.tool, parsed.args)
        yield ToolExecuted(turn=turn, tool=parsed.tool, args=parsed.args, result=result)
        observation = result.output

    yield MaxTurnsHit(turn=max_turns)
