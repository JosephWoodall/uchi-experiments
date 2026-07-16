"""Additional grounding/verification signals for Ducky, moving toward
uchi's actual mission (verify against something real, abstain otherwise)
while keeping next-token prediction as Ducky's core objective -- these are
added signals predict_next (or a caller) can consult, not a replacement of
the graph/confidence mechanism already in inference.py.
"""
import ast

import torch
import torch.nn.functional as F


def verify_code_syntax(text: str) -> bool:
    """Does this parse as valid Python? Scoped down from uchi's actual
    REPL/TDD (execute and check correctness) to parse-validity -- generated
    snippets are rarely complete runnable programs, so "does it execute
    correctly" isn't well-defined here, but "is this even syntactically
    real code" is a genuine grounding check, not a proxy for one. uchi's
    Empirical Grounding axiom is "prove it in a live sandbox before
    speaking"; this is the cheapest honest version of that for snippets
    that can't be run standalone.
    """
    try:
        ast.parse(text)
        return True
    except SyntaxError:
        return False


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


def identifier_grounded(decoded_text: str, symbol_table: set) -> bool:
    """Does a generated identifier-like span match a symbol actually seen
    in the corpus? A stricter, string-level check than the token-level AST
    fact edges."""
    words = [w.strip("()[]{}:,.\"'") for w in decoded_text.strip().split()]
    words = [w for w in words if w]
    if not words:
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
