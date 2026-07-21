"""Additional grounding/verification signals for Ducky, moving toward
uchi's actual mission (verify against something real, abstain otherwise)
while keeping next-token prediction as Ducky's core objective -- these are
added signals predict_next (or a caller) can consult, not a replacement of
the graph/confidence mechanism already in inference.py.
"""
import ast
import operator
import re
import signal
from typing import Optional

import torch
import torch.nn.functional as F

# Restricted builtins allowlist for run_sandboxed -- no os/subprocess/open/
# import, so an executed generation can't touch the filesystem or spawn
# processes. Ported from bench_ducky.py's own already-validated
# `_SAFE_BUILTINS` (canned-correct=100%, wrong answer fails, infinite loop
# times out) -- moved here so it's reusable as a generation-time grounding
# signal, not just an offline benchmark's grading mechanism.
_SAFE_BUILTINS = {
    "len": len, "range": range, "sum": sum, "min": min, "max": max, "abs": abs,
    "round": round, "sorted": sorted, "list": list, "dict": dict, "set": set,
    "tuple": tuple, "str": str, "int": int, "float": float, "bool": bool,
    "enumerate": enumerate, "zip": zip, "map": map, "filter": filter,
    "isinstance": isinstance, "ValueError": ValueError, "TypeError": TypeError,
    "ZeroDivisionError": ZeroDivisionError, "True": True, "False": False, "None": None,
}


class _SandboxTimeout(Exception):
    pass


def _sandbox_alarm_handler(signum, frame):
    raise _SandboxTimeout()


def run_sandboxed(code: str, extra_statements: list = None, timeout_s: float = 2.0) -> dict:
    """Execute *code* (then, if given, each of *extra_statements* -- e.g.
    real asserts) in a namespace restricted to _SAFE_BUILTINS, under a
    SIGALRM wall-clock timeout. Generalizes bench_ducky.py's `run_task`
    (extra_statements replaces its hardcoded `asserts` param, defaulting to
    none) so the same validated safety primitive is reusable as a plain
    "does this even run" check, not just assert-based grading.
    """
    safe_globals = {"__builtins__": dict(_SAFE_BUILTINS)}
    local_ns: dict = {}
    old_handler = signal.signal(signal.SIGALRM, _sandbox_alarm_handler)
    signal.alarm(max(1, int(timeout_s)))
    try:
        exec(code, safe_globals, local_ns)
        for stmt in (extra_statements or []):
            exec(stmt, safe_globals, local_ns)
        return {"passed": True}
    except _SandboxTimeout:
        return {"passed": False, "error": "timeout"}
    except Exception as e:
        return {"passed": False, "error": f"{type(e).__name__}: {e}"}
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, old_handler)


def verify_code_syntax(text: str) -> bool:
    """Does this parse as valid Python? Scoped down from uchi's actual
    REPL/TDD (execute and check correctness) to parse-validity -- generated
    snippets are rarely complete runnable programs, so "does it execute
    correctly" isn't well-defined here, but "is this even syntactically
    real code" is a genuine grounding check, not a proxy for one. uchi's
    Empirical Grounding axiom is "prove it in a live sandbox before
    speaking"; this is the cheapest honest version of that for snippets
    that can't be run standalone.

    Real, honest gap, found by run_flywheel.py's round-1 output: this
    passes on a function body that's nothing but a comment (`# whatever
    gibberish`), since a `#` line produces no AST node at all and isn't a
    syntax error -- "parses" is not the same as "contains real code."
    has_real_statement below is the fix for callers (like the flywheel)
    that need that stronger guarantee.
    """
    try:
        ast.parse(text)
        return True
    except SyntaxError:
        return False


def executes_without_error(code: str, timeout_s: float = 2.0) -> Optional[bool]:
    """The real behavioral next rung beyond verify_code_syntax: does this
    actually run without raising, in the same restricted-builtins +
    SIGALRM-timeout sandbox bench_ducky.py already validated (run_sandboxed
    above) -- stronger than "parses," weaker than bench_ducky.py's full
    assert-based grading (no ground-truth assertions exist at generation
    time, only "did defining/running this blow up"). Returns None if the
    code doesn't even parse (verify_code_syntax fails first -- nothing
    meaningful to execute), True/False otherwise.
    """
    if not verify_code_syntax(code):
        return None
    return run_sandboxed(code, timeout_s=timeout_s)["passed"]


def is_complete_statement(prompt: str, completion_so_far: str) -> bool:
    """Cheap, checkable stopping signal for generation: the model has just
    emitted a blank line (a real, common Python convention marking the end
    of a top-level block, like a function) AND the combined text already
    parses as valid Python. Neither condition alone is a real signal --
    "return x\\n" alone parses fine but isn't necessarily where anyone
    meant to stop; a blank line alone doesn't mean what came before it was
    even valid syntax -- together they're a checkable proxy for "this
    looks done", not a guess about the future. Motivated directly by
    tasks/ducky.md's "minimum viable size" round: several real failures
    were generation drifting past the function's natural end into an
    unrelated new `def`, always burning the full max_new_tokens budget
    regardless of whether the answer was already complete.
    """
    if not completion_so_far.endswith("\n\n"):
        return False
    try:
        ast.parse(prompt + completion_so_far)
        return True
    except SyntaxError:
        return False


def has_real_statement(text: str) -> bool:
    """Stronger than verify_code_syntax: does the (first) function
    definition's body contain at least one statement beyond its own
    docstring -- rules out completions that parse only because they
    degenerated into a comment (no AST node at all) or a bare `pass`.
    Found necessary directly: run_flywheel.py's naive
    verify_code_syntax-only gate let 6/6 "verified" Tier-2 examples
    through that were pure comment noise after the docstring, not real
    code -- this is the actual fix, not a hypothetical hardening.
    """
    try:
        tree = ast.parse(text)
    except SyntaxError:
        return False
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef):
            body = node.body
            if body and isinstance(body[0], ast.Expr) and isinstance(body[0].value, ast.Constant) \
                    and isinstance(body[0].value.value, str):
                body = body[1:]  # skip the docstring itself
            real = [s for s in body if not isinstance(s, ast.Pass)]
            return len(real) > 0
    return False


_ALLOWED_BINOPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.Pow: operator.pow,
}
_ALLOWED_UNARYOPS = {
    ast.USub: operator.neg,
    ast.UAdd: operator.pos,
}


def _eval_arithmetic_node(node):
    if isinstance(node, ast.Constant):
        if isinstance(node.value, (int, float)) and not isinstance(node.value, bool):
            return node.value
        raise ValueError(f"non-numeric constant: {node.value!r}")
    if isinstance(node, ast.BinOp) and type(node.op) in _ALLOWED_BINOPS:
        return _ALLOWED_BINOPS[type(node.op)](
            _eval_arithmetic_node(node.left), _eval_arithmetic_node(node.right)
        )
    if isinstance(node, ast.UnaryOp) and type(node.op) in _ALLOWED_UNARYOPS:
        return _ALLOWED_UNARYOPS[type(node.op)](_eval_arithmetic_node(node.operand))
    raise ValueError(f"disallowed node: {type(node).__name__}")


def evaluate_arithmetic(expr: str) -> Optional[float]:
    """Real arithmetic, not mimicry: a restricted AST walker, never `eval()`.
    Whitelists only numeric constants, +-*/**, and unary +/- -- anything
    else (Name, Call, Attribute, Subscript, Compare, ...) raises internally
    and is caught below, returning None (graceful abstention, same
    convention as check_call_arity_consistency's `consistent=None`) rather
    than executing arbitrary code. Deliberately narrower than uchi's
    subprocess-sandboxed run_python tool (see scratchpad.py in the uchi
    project) -- arithmetic doesn't need arbitrary-code generality, a
    subprocess, or a new external dependency; a whitelisted walker is
    safer and simpler to verify for this one narrow job.
    """
    try:
        tree = ast.parse(expr, mode="eval")
        return _eval_arithmetic_node(tree.body)
    except (SyntaxError, ValueError, TypeError, ZeroDivisionError, OverflowError):
        return None


_ARITH_EXPR_RE = re.compile(
    r"(-?\d+(?:\.\d+)?(?:\s*[+\-*/]\s*-?\d+(?:\.\d+)?)+)\s*=\s*$"
)
_ARITH_CLAIM_RE = re.compile(
    r"(-?\d+(?:\.\d+)?(?:\s*[+\-*/]\s*-?\d+(?:\.\d+)?)+)\s*=\s*(-?\d+(?:\.\d+)?)"
)


def find_arithmetic_expression(text: str, tail_chars: int = 64) -> Optional[str]:
    """Detects a just-completed literal-only arithmetic expression sitting
    right at the end of *text*, immediately before an '=' the model has
    just emitted (e.g. "...12 + 5 * 3 - 4 =") -- the trigger point for
    generate_with_calculator to stop mimicking and compute for real.
    Scoped to the tail (default last 64 chars), not the whole text, so a
    stale expression earlier in a multi-step trace never re-triggers.
    """
    tail = text[-tail_chars:]
    m = _ARITH_EXPR_RE.search(tail)
    return m.group(1).strip() if m else None


def arithmetic_grounded(text: str) -> Optional[bool]:
    """Post-hoc verification signal (the cheap first rung, before the real
    fix): finds every "expr = claimed_value" pattern in *text* and checks
    the claimed value against evaluate_arithmetic's real result. Returns
    None if no checkable claim exists at all (abstain-neutral, same
    convention as check_call_arity_consistency's untestable case), True
    only if every claim found is correct, False if any is wrong.
    """
    matches = list(_ARITH_CLAIM_RE.finditer(text))
    if not matches:
        return None
    checked_any = False
    for m in matches:
        real = evaluate_arithmetic(m.group(1))
        if real is None:
            continue
        checked_any = True
        if abs(real - float(m.group(2))) > 1e-6:
            return False
    return True if checked_any else None


@torch.no_grad()
def self_critique_score(model, prompt_ids: torch.Tensor, generated_ids: list) -> float:
    """Re-score the model's own just-generated tokens in teacher-forced
    mode -- does the model, on reflection, confidently stand by what it
    produced, rather than the sampling-time confidence it used to pick
    those tokens? Single-model analog of uchi's Devil's Advocate: no
    second expert needed, just re-evaluate the same model against its own
    output.
    """
    gen_t = torch.tensor([generated_ids], dtype=torch.long)
    full = torch.cat([prompt_ids, gen_t], dim=1)
    full = full[:, -model.cfg.block_size :]
    n_gen = min(len(generated_ids), full.size(1) - 1)
    logits, _, _, _ = model(full)
    gen_logits = logits[0, -n_gen - 1 : -1, :]
    probs = F.softmax(gen_logits, dim=-1)
    gen_ids_t = torch.tensor(generated_ids[-n_gen:])
    confidences = probs[torch.arange(n_gen), gen_ids_t]
    return confidences.mean().item()


def build_symbol_table(code_source: str) -> set:
    """Real identifiers (function names, attribute names, imported names)
    as whole strings -- grounds facts at the level meaning actually lives
    at, not BPE-token fragments. graph.py's AST-fact edges hit a precision
    ceiling because rare identifiers fragment to near-character-level
    tokens ('self' -> 'break'); checking a *decoded* identifier string
    against a real symbol table sidesteps that instead of chasing better
    token-boundary alignment.
    """
    tree = ast.parse(code_source)
    symbols = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef):
            symbols.add(node.name)
        elif isinstance(node, ast.Attribute):
            symbols.add(node.attr)
        elif isinstance(node, (ast.Import, ast.ImportFrom)):
            for alias in node.names:
                symbols.add(alias.name.split(".")[-1])
        elif isinstance(node, ast.Name):
            symbols.add(node.id)
    return symbols


def identifier_grounded(decoded_text: str, symbol_table: set, min_length: int = 3) -> bool:
    """Does a generated identifier-like span match a symbol actually seen
    in the corpus? A stricter, string-level check than the token-level AST
    fact edges.

    min_length=3 found necessary directly, not assumed: tested against the
    real symbol table built from corpus_core.txt (22,537 identifiers) and
    single-letter names (x, a, n, ...) are present near-universally --
    real, common loop/param variable names, not rare or specific ones. Any
    generated text ending in a common single letter was trivially
    "grounded" regardless of whether anything meaningful was actually
    verified (identifier_grounded("return x", ...) was True), while a
    genuinely novel identifier like "flatten_one_level" correctly fails --
    the check is well-calibrated for specific/rare identifiers and near-
    meaningless for generic short ones. Filtering below min_length doesn't
    make short identifiers "wrong," just declines to credit them as
    grounded evidence, matching this file's existing discipline of only
    claiming a signal means what it says.
    """
    words = [w.strip("()[]{}:,.\"'") for w in decoded_text.strip().split()]
    words = [w for w in words if w]
    if not words or len(words[-1]) < min_length:
        return False
    return words[-1] in symbol_table


def check_call_arity_consistency(text: str) -> dict:
    """Narrow, deterministic self-consistency veto -- modeled on uchi's
    relational_reasoning.py, which handles one specific, well-defined
    transitive-relation class (comparatives) rather than attempting
    general logical inference. This handles one specific, well-defined
    class for code: does the same function name get called with the same
    number of positional arguments everywhere in this text? Only plain
    `name(...)` calls are checked (not `obj.method(...)`, which would need
    real type inference to resolve which `method` is even being called) --
    same "stay narrow, stay correct" discipline as the relation it's
    modeled on.

    Same additive-only invariant as relational_reasoning.py: this can only
    flag an inconsistency, never confirm correctness (matching arities
    everywhere doesn't mean the code is right, just that it isn't
    self-contradictory on this one narrow axis) -- and it abstains
    (consistent=None) rather than guess when the text doesn't parse at
    all, since arity can't be checked without a real AST.

    Real, honest caveat: Python allows default arguments and *args, so a
    genuinely correct program can legitimately call the same function with
    different argument counts. This is a heuristic that will sometimes
    flag valid code, same kind of narrow-on-purpose tradeoff
    relational_reasoning.py accepts for its own pattern-matched relations.
    """
    try:
        tree = ast.parse(text)
    except SyntaxError:
        return {"consistent": None, "reason": "syntax invalid, cannot check"}

    arities: dict = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
            arities.setdefault(node.func.id, set()).add(len(node.args))

    conflicts = [{"name": name, "arities": sorted(counts)}
                 for name, counts in arities.items() if len(counts) > 1]
    return {"consistent": len(conflicts) == 0, "conflicts": conflicts}


def build_ngram_index(ids, n: int = 4) -> set:
    """Verbatim n-grams seen in the actual source -- a cheap stand-in for
    brain.uchi's retrieval index: not embeddings/similarity search, just
    "did this exact sequence occur in real source material." A match is a
    grounding signal (recitation of something real, not fabrication); a
    novel n-gram isn't necessarily wrong -- for rj it might just be a fine
    creative continuation -- but it isn't grounded in the retrieval sense,
    so it shouldn't be scored as verified.
    """
    ids = ids.tolist() if hasattr(ids, "tolist") else list(ids)
    return {tuple(ids[i : i + n]) for i in range(len(ids) - n + 1)}


def ngram_grounded(token_sequence: list, ngram_index: set, n: int = 4) -> bool:
    if len(token_sequence) < n:
        return False
    return tuple(token_sequence[-n:]) in ngram_index
