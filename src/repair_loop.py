"""generate_with_repair: ReAct-lite sequential retry loop, modeled on
uchi's agentic_repair.py -- the key difference from reject-and-resample
(inference.py) is that this is SEQUENTIAL and INFORMED, not parallel and
blind. Reject-and-resample fires N independent candidates that never see
each other; this generates one, and if it fails, splices the actual
failure text (the real SyntaxError) into the next attempt's prompt before
retrying -- attempt 2 knows *why* attempt 1 failed.

Ducky has no sandboxed execution (unlike uchi's ExecutionSandbox), so the
only checkable failure mode here is static: syntax validity and self-
critique confidence, not real test-pass/fail. Tri-state outcome, same
shape as uchi's Outcome.PASS/FAIL/ABSTAIN: PASS (checks cleared), FAIL
(produced something on every attempt, never cleared the bar), ABSTAIN
(never produced any tokens at all, on any attempt -- the model had nothing
to say, distinct from "said something wrong"). max_attempts defaults to 4,
matching uchi's own retry cap.
"""
import ast

from grounding import identifier_grounded, self_critique_score, verify_code_syntax
from inference import generate_with_grounding


class Outcome:
    PASS = "pass"
    FAIL = "fail"
    ABSTAIN = "abstain"


def _syntax_error_text(full_text: str) -> str:
    try:
        ast.parse(full_text)
        return ""
    except SyntaxError as e:
        # e.msg often already includes its own "(detected at line X)" for
        # errors like unterminated strings, which refers to where the
        # parser gave up, not e.lineno (where the offending token started)
        # -- appending both was printing two different, confusing line
        # numbers for the same error. e.msg alone is the correct, complete
        # message; CPython already puts position info in it when relevant.
        return f"{type(e).__name__}: {e.msg}"


def generate_with_repair(model, tok, prompt: str, max_new_tokens: int, domain: str,
                          temperature: float = 0.0, top_p: float = 1.0, repetition_penalty: float = 1.0,
                          symbol_table: set = None, ngram_index: set = None, ngram_n: int = 4,
                          max_attempts: int = 4, min_self_critique: float = 0.0):
    attempts_log = []
    current_prompt = prompt
    result = None

    for attempt in range(1, max_attempts + 1):
        result = generate_with_grounding(model, tok, current_prompt, max_new_tokens, domain,
                                          temperature=temperature, top_p=top_p,
                                          repetition_penalty=repetition_penalty,
                                          symbol_table=symbol_table,
                                          ngram_index=ngram_index, ngram_n=ngram_n)
        attempts_log.append({
            "attempt": attempt, "prompt": current_prompt,
            "generated_text": result["generated_text"],
            "n_tokens_generated": result["n_tokens_generated"],
            "syntax_valid": result.get("syntax_valid"),
            "self_critique_score": result.get("self_critique_score"),
        })

        if result["n_tokens_generated"] == 0:
            continue  # nothing generated -- no new failure info to splice in, just retry

        ok = (domain != "code" or result.get("syntax_valid", True)) and \
             result.get("self_critique_score", 0.0) >= min_self_critique
        if ok:
            result["outcome"] = Outcome.PASS
            result["attempts_log"] = attempts_log
            result["n_attempts"] = attempt
            # generated_text stays relative to the ORIGINAL prompt for a
            # passing result on attempt 1; for a later attempt it's
            # relative to current_prompt, which already includes the
            # original prompt as a prefix, so this is still correct text.
            return result

        if domain == "code":
            full_text = current_prompt + result["generated_text"]
            err = _syntax_error_text(full_text)
            feedback = (f"\n# Previous attempt had an error: {err}\n" if err else
                        "\n# Previous attempt was syntactically valid but low-confidence; try again.\n")
        else:
            feedback = "\n# Previous attempt was low-confidence; try again.\n"
        current_prompt = prompt + feedback

    outcome = Outcome.ABSTAIN if (result is None or attempts_log[-1]["n_tokens_generated"] == 0) else Outcome.FAIL
    result["outcome"] = outcome
    result["attempts_log"] = attempts_log
    result["n_attempts"] = len(attempts_log)
    return result
