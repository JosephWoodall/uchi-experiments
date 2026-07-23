"""Single-model inference -- pure neural prediction now, no external signal
mixed into the logits. Two things this file used to do were both removed
this session (2026-07-22), both by the user's explicit call, not bugs:

1. Uncertainty-driven abstention (swarm.md post-mortem concept 5): measured
   as a net positive on selective-prediction accuracy (Phase H), but
   refusing to answer was judged to stifle Ducky's output more than that
   accuracy gain was worth. predict_next physically cannot return "no
   token" anymore.
2. TokenGraph blending (swarm.md concept 6): the fast/slow path split used
   to mean "skip the graph when confident, blend it into the logits when
   not" -- alpha*neural_logits + beta*graph_scores. Removed because it was
   judged to be causing bad results: a fixed statistical co-occurrence
   table nudging every uncertain token, independent of what the neural
   network itself actually predicted. Every token is now 100% the model's
   own computation, no fast/slow distinction left to make, no graph
   argument anywhere in this file anymore.
"""
import torch
import torch.nn.functional as F

from grounding import (
    check_call_arity_consistency,
    evaluate_arithmetic,
    executes_without_error,
    find_arithmetic_expression,
    identifier_grounded,
    is_complete_statement,
    ngram_grounded,
    self_critique_score,
    verify_code_syntax,
)
from session_memory import SessionTrie


def _choose(probs: torch.Tensor, temperature: float, top_p: float = 1.0) -> int:
    """temperature=0 -> argmax (deterministic, the original behavior).
    temperature>0 -> multinomial sample, tempered.

    top_p<1.0 (nucleus sampling, Holtzman et al. 2019 arXiv:1904.09751 --
    the same paper this file's no-repeat-ngram fix already cites) restricts
    sampling to the smallest set of tokens whose cumulative probability
    covers top_p, zeroing out the long low-probability tail before
    sampling. Added because raw temperature=0.8 alone, once generation
    started running long (300 tokens, not the old short abstain-truncated
    completions), let through enough individually-low-probability tokens
    to produce garbled words ("shadn", "wcup") -- diverse but not fluent.
    top_p=1.0 (default) is a no-op, identical to the pre-existing behavior.
    """
    if temperature <= 0:
        return probs.argmax(dim=-1).item()
    tempered = torch.softmax(torch.log(probs + 1e-12) / temperature, dim=-1)
    if top_p < 1.0:
        sorted_probs, sorted_idx = torch.sort(tempered, descending=True)
        cumulative = torch.cumsum(sorted_probs, dim=-1)
        # Keep the smallest prefix whose cumulative mass >= top_p, always
        # keeping at least the single most-probable token even if its own
        # probability already exceeds top_p.
        cutoff = (cumulative <= top_p).sum().item()
        cutoff = max(cutoff, 1)
        mask = torch.zeros_like(tempered, dtype=torch.bool)
        mask[sorted_idx[:cutoff]] = True
        tempered = torch.where(mask, tempered, torch.zeros_like(tempered))
        tempered = tempered / tempered.sum()
    return torch.multinomial(tempered, num_samples=1).item()


def _block_repeat_ngrams(logits: torch.Tensor, context: list, n: int) -> torch.Tensor:
    """No-repeat-ngram blocking (standard decoding technique, e.g.
    HuggingFace's no_repeat_ngram_size): masks out (logit=-inf) any
    candidate token that would recreate an n-gram already generated
    earlier in this same context. This is the actual fix for the
    self-reinforcing repetition loop (Holtzman et al. 2019,
    arXiv:1904.09751): measured directly on this checkpoint, confidence in
    repeating a phrase climbs every cycle (0.54 -> 0.82 -> 0.88 -> 0.90 at
    matched positions across cycles), converging toward ~0.97 -- nothing
    based on confidence alone can break out of that once it starts, since
    confidence is *increasing*, not decreasing. Only directly forbidding
    the repeat works. Applied before confidence/fast-path is computed, not
    after, so the rest of the abstention machinery naturally reconsiders
    whatever's left once the repeat is removed, rather than needing a
    separate override path.
    """
    if len(context) < n - 1:
        return logits
    seen_ngrams: dict = {}
    for i in range(len(context) - n + 1):
        prefix = tuple(context[i:i + n - 1])
        seen_ngrams.setdefault(prefix, set()).add(context[i + n - 1])
    current_prefix = tuple(context[-(n - 1):])
    blocked = seen_ngrams.get(current_prefix)
    if not blocked:
        return logits
    logits = logits.clone()
    for tok in blocked:
        logits[tok] = float("-inf")
    return logits


def _apply_repetition_penalty(logits: torch.Tensor, context: list, penalty: float,
                                recency_decay: float = 1.0) -> torch.Tensor:
    """CTRL-style repetition penalty (Keskar et al. 2019, arXiv:1909.05858),
    with an optional recency-decay extension that was tried and measurably
    REJECTED as a default -- kept as an off-by-default parameter for
    experimentation, not because it helped.

    effective_penalty(tok) = 1 + (penalty - 1) * recency_decay**(distance - 1)
    where distance = tokens since tok's most recent occurrence (1 =
    immediately preceding token). recency_decay=1.0 (default) means every
    occurrence gets the full flat `penalty` regardless of distance --
    identical to the original, validated behavior. recency_decay<1.0 lets
    the penalty fade for tokens seen further back.

    The hypothesis for adding decay was real: a flat penalty treats a
    necessary common word ("I", "not", "thou") used 200 tokens ago
    identically to one used 2 tokens ago, which could suppress ordinary
    function words at long context. **Measured directly (5-seed,
    250-token distinct-2/distinct-3 diversity test) and found to make
    things WORSE, monotonically, across the whole tested range**:
    decay=1.0 scored distinct-2=0.80/distinct-3=0.94 (best); decay=0.9
    scored 0.55/0.80 (worst). Letting old tokens' penalties fade doesn't
    selectively free up necessary common words -- it just as freely lets
    old REPEATED PHRASES resurface too, and at this model's scale that
    dominates. A real, informative negative result, not adopted.

    Targets a different failure mode than _block_repeat_ngrams: that
    function only stops an EXACT repeated n-gram; it does nothing about
    the same handful of tokens recurring across many short phrases that
    are each individually novel (never literally repeat the same 4-gram).
    penalty=1.0 (default) is a no-op, identical to no penalty at all.
    """
    if penalty == 1.0:
        return logits
    logits = logits.clone()
    n = len(context)
    last_seen: dict = {}
    for i, tok in enumerate(context):
        last_seen[tok] = i  # overwritten on each occurrence -> ends up as the LAST (most recent) index
    for tok, idx in last_seen.items():
        distance = n - idx
        effective_penalty = 1.0 + (penalty - 1.0) * (recency_decay ** (distance - 1))
        logits[tok] = logits[tok] / effective_penalty if logits[tok] > 0 else logits[tok] * effective_penalty
    return logits


def _apply_session_memory(session_memory: SessionTrie, context: list, chosen_token: int, info: dict) -> None:
    """session_memory is the working-memory trie (session_memory.py) --
    distinct from n-gram grounding, which checks against the training
    corpus; this checks against what THIS generation has already
    produced. If the
    exact trailing context has come up before in this same generation,
    session_consistent=False flags that a different token got chosen this
    time for the identical context -- self-contradiction, not corpus
    disagreement. Always records the observation afterward regardless of
    the consistency outcome, so future steps in this generation can check
    against it too.
    """
    if session_memory is None:
        return
    prior = session_memory.lookup(context)
    if prior is not None:
        info["session_consistent"] = chosen_token in prior
    session_memory.observe(context, chosen_token)


class ChunkedState:
    """Carries RWKV state across a generation session so a hybrid model
    isn't limited to only ever seeing the last block_size tokens -- fixes
    a real gap: predict_next previously always did
    `model(idx[:, -block_size:])`, a fresh forward pass over just the
    trailing window, every single step, discarding everything before it
    and never calling model.py's own documented rwkv_states mechanism
    (hidden_states' own docstring: "continue a recurrence across chunks
    -- the unlimited-context mechanism"). The RWKV layers' O(1)-state
    carry was structurally verified elsewhere in this project but never
    actually exploited by Ducky's live generation loop until now.

    Position embeddings reset to 0 at the start of each chunk (required,
    not just convenient: pos_emb is `nn.Embedding(block_size, d_model)`,
    a table with exactly block_size rows -- an absolute position past
    block_size-1 would be an out-of-bounds lookup). The attention layer
    (if any, via attention_layers) only ever sees the current chunk, same
    as before -- it has no state to carry, by construction. The RWKV
    layers get the compressed memory of everything before the current
    chunk via carried_states; that's the whole unlimited-context
    property, now actually used at inference time, not just proven to
    exist in an isolated test.

    Cost profile: no worse than the old crop-and-forward on average --
    the current chunk grows from 1 token up to block_size before
    resetting, so the average per-step forward cost is *lower* than
    always reprocessing a full block_size window, plus one extra
    full-chunk forward pass at each block_size boundary to seal in the
    correct carried_states before resetting.
    """
    def __init__(self):
        self.chunk: list = []
        self.carried_states = None

    @torch.no_grad()
    def prime(self, model, prompt_ids: list):
        """Pre-processes a prompt longer than block_size in full chunks,
        building up carried_states for everything except the final
        (always length >= 1, never fully drained -- see append()) active
        chunk. A prompt whose length is an exact multiple of block_size
        still leaves at least 1 token active, not 0: next_logits() needs
        at least one real token to produce logits from, and carried_states
        alone (with zero active tokens) can't supply that.
        """
        block_size = model.cfg.block_size
        states = None
        i = 0
        while len(prompt_ids) - i > block_size:
            chunk = prompt_ids[i : i + block_size]
            _, _, _, states = model(torch.tensor([chunk], dtype=torch.long), states)
            i += block_size
        self.chunk = list(prompt_ids[i:])
        self.carried_states = states

    @torch.no_grad()
    def next_logits(self, model) -> torch.Tensor:
        """Forward pass over the current chunk (with carried_states as the
        RWKV layers' starting state), returning next-token logits. Caller
        appends the sampled token via .append(); rollover happens
        automatically once the buffer would exceed block_size.
        """
        chunk_t = torch.tensor([self.chunk], dtype=torch.long)
        logits, _, _, _ = model(chunk_t, self.carried_states)
        return logits[0, -1, :]

    @torch.no_grad()
    def append(self, model, token_id: int) -> None:
        self.chunk.append(token_id)
        block_size = model.cfg.block_size
        if len(self.chunk) > block_size:
            # Fold everything except the newest token into carried_states --
            # NOT the whole (now block_size+1-long) chunk. Two things this
            # avoids: (1) forward-passing more than block_size tokens at
            # once, which would need a position index past pos_emb's valid
            # [0, block_size-1] range; (2) leaving the active chunk fully
            # empty, which next_logits() can't produce anything from. The
            # newest token stays active and gets folded into carried_states
            # on a LATER call, exactly once, never duplicated or dropped.
            to_fold, keep = self.chunk[:-1], self.chunk[-1]
            _, _, _, self.carried_states = model(torch.tensor([to_fold], dtype=torch.long), self.carried_states)
            self.chunk = [keep]


def predict_next(model, idx: torch.Tensor, chunk_state: "ChunkedState" = None,
                  ngram_index: set = None, ngram_n: int = 4, temperature: float = 0.0, top_p: float = 1.0,
                  session_memory: SessionTrie = None, no_repeat_ngram_size: int = 4,
                  repetition_penalty: float = 1.0, repetition_penalty_decay: float = 1.0):
    """Returns (token_id, info_dict) for the next token after idx -- pure
    neural prediction now. Graph-blending removed (2026-07-22, user's
    explicit call: "that's causing Ducky to give bad results"): the
    TokenGraph co-occurrence lookup used to get mixed directly into the
    logits whenever neural confidence was low (`combined_logits = alpha *
    neural_logits + beta * graph_scores`) -- a fixed statistical table
    from the training corpus nudging every uncertain token, not just the
    neural network's own computation. That's gone; every token now comes
    from the model alone, exactly like a typical LLM's forward pass.
    fast_threshold/calibrate_thresholds and the graph parameter are gone
    with it -- there's no second path left to route between.

    ngram_index (from grounding.build_ngram_index), if given, is still
    computed and reported in info (real evidence this exact sequence
    occurred in real source) -- observability only, never affects which
    token gets returned.

    session_memory (session_memory.SessionTrie), if given, is consulted
    and updated on every call -- see _apply_session_memory. Observability
    only: it annotates info, it does not override the chosen token.

    temperature=0 (default) is fully deterministic -- the original
    behavior. temperature>0 samples (optionally restricted to top_p's
    nucleus), so repeated calls on the same prompt can genuinely differ.

    no_repeat_ngram_size (default 4): blocks any candidate that would
    recreate an n-gram already generated in this context -- see
    _block_repeat_ngrams. Set to 0 to disable.

    repetition_penalty (default 1.0, no-op): soft per-token penalty on
    anything already seen in context -- see _apply_repetition_penalty.
    Addresses repeated *phrasing* (the same tokens recurring across many
    non-identical short phrases), which no_repeat_ngram_size alone doesn't
    catch since it only blocks exact repeated n-grams. repetition_penalty_
    decay=1.0 (default, flat -- see _apply_repetition_penalty for why the
    recency-weighted alternative was tested and rejected, not just unused).

    chunk_state (ChunkedState, default None): if given, next-token logits
    come from the chunked, RWKV-state-carrying path instead of a fresh
    crop-and-forward over just the last block_size tokens -- see
    ChunkedState's own docstring for why this actually removes the
    128-token effective context limit, not just the max_new_tokens output
    length (which was never limited). idx is still used, regardless of
    chunk_state, for the context-dependent checks below (no-repeat-ngram,
    repetition penalty, n-gram grounding, session memory) -- those operate
    on the real full token history either way.
    """
    with torch.no_grad():
        if chunk_state is not None:
            neural_logits = chunk_state.next_logits(model)
        else:
            logits, _, _, _ = model(idx[:, -model.cfg.block_size :])
            neural_logits = logits[0, -1, :]
        if no_repeat_ngram_size > 0:
            neural_logits = _block_repeat_ngrams(neural_logits, idx[0].tolist(), no_repeat_ngram_size)
        if repetition_penalty != 1.0:
            neural_logits = _apply_repetition_penalty(neural_logits, idx[0].tolist(), repetition_penalty,
                                                       repetition_penalty_decay)
        neural_probs = F.softmax(neural_logits, dim=-1)
        neural_conf = neural_probs.max(dim=-1).values.item()
        neural_pred = _choose(neural_probs, temperature, top_p)
        if chunk_state is not None:
            chunk_state.append(model, neural_pred)

    info = {"confidence": round(neural_conf, 4)}
    if ngram_index is not None:
        info["ngram_grounded"] = ngram_grounded(idx[0].tolist() + [neural_pred], ngram_index, ngram_n)
    _apply_session_memory(session_memory, idx[0].tolist(), neural_pred, info)
    return neural_pred, info


def generate_with_grounding(model, tok, prompt: str, max_new_tokens: int, domain: str,
                             temperature: float = 0.0, top_p: float = 1.0, repetition_penalty: float = 1.0,
                             symbol_table: set = None, ngram_index: set = None, ngram_n: int = 4,
                             stop_when_complete: bool = True, use_chunked_state: bool = True):
    """Sequence-level wrapper: predict_next per token (pure neural now --
    abstention and graph-blending were both removed), then post-hoc
    verification on the completed span. Syntax validity, self-critique,
    and identifier grounding are inherently sequence/span-level checks
    (there's no "is this syntactically valid" for a single token) --
    n-gram grounding is the one signal granular enough to fold into
    predict_next's own per-token info, done there instead of here.

    temperature=0.0 (default) is deterministic argmax. temperature>0
    samples (optionally nucleus-restricted via top_p) -- matters once
    generation runs long (300 tokens, not the old short abstain-truncated
    completions): greedy argmax over a long span is what drives the
    repetitive degeneration no_repeat_ngram_size alone doesn't prevent.

    use_chunked_state=True (default): uses ChunkedState so the RWKV
    layers carry compressed memory across block_size chunk boundaries
    instead of every step re-forwarding a fresh, hard-cropped
    last-block_size-tokens window -- see ChunkedState's own docstring.
    Fixes a real limit: past ~block_size tokens (128 for typical
    checkpoints), the old path couldn't see the earlier part of its own
    generation at all, which measurably contributed to degradation past
    ~150-200 tokens. Set False to restore the old crop-and-forward
    behavior exactly.

    stop_when_complete (code domain only, via grounding.is_complete_statement):
    stop as soon as the completion looks like a finished statement (parses
    + a blank line just emitted) instead of always spending the full
    max_new_tokens budget -- motivated directly by generations that were
    otherwise observed drifting past a function's natural end into an
    unrelated new `def`.
    """
    prompt_ids = tok.encode(prompt)
    ids = torch.tensor([prompt_ids], dtype=torch.long)
    prompt_len = ids.size(1)
    generated = []
    stopped_complete = False
    # Fresh per call, discarded when this returns -- session-scoped, not
    # persisted (see session_memory.py). Seeding it with the prompt's own
    # tokens would double-count the prompt as "already generated"; it
    # starts empty and only accumulates tokens this call actually produces.
    session_memory = SessionTrie()
    chunk_state = None
    if use_chunked_state:
        chunk_state = ChunkedState()
        chunk_state.prime(model, prompt_ids)

    for _ in range(max_new_tokens):
        next_id, info = predict_next(model, ids, chunk_state=chunk_state, temperature=temperature, top_p=top_p,
                                      repetition_penalty=repetition_penalty,
                                      ngram_index=ngram_index, ngram_n=ngram_n,
                                      session_memory=session_memory)
        generated.append(next_id)
        ids = torch.cat([ids, torch.tensor([[next_id]])], dim=1)
        if domain == "code" and stop_when_complete and is_complete_statement(prompt, tok.decode(generated)):
            stopped_complete = True
            break

    text = tok.decode(generated) if generated else ""
    result = {"prompt": prompt, "generated_text": text, "n_tokens_generated": len(generated),
              "stopped_complete": stopped_complete}

    if generated:
        result["self_critique_score"] = self_critique_score(model, ids[:, :prompt_len], generated)
        if domain == "code":
            full_text = prompt + text
            result["syntax_valid"] = verify_code_syntax(full_text)
            executes = executes_without_error(full_text)
            if executes is not None:
                result["executes"] = executes
            if symbol_table is not None:
                result["identifier_grounded"] = identifier_grounded(full_text, symbol_table)
            arity_check = check_call_arity_consistency(full_text)
            if arity_check["consistent"] is not None:
                result["arity_consistent"] = arity_check["consistent"]
                if arity_check["conflicts"]:
                    result["arity_conflicts"] = arity_check["conflicts"]

    return result


def _maybe_splice_arithmetic(tok, ids: torch.Tensor, generated: list):
    """Checks the tail of the full running context (prompt + generated so
    far -- an expression can span that boundary or sit entirely within
    either side) for a just-completed arithmetic expression right before
    an '='. If found and evaluable, appends the real computed value's
    tokens to *generated* and returns (new_ids, splice_record); returns
    (ids, None) unchanged otherwise. Pulled out of
    generate_with_calculator's loop so it's directly unit-testable against
    a hand-built context, independent of whatever a real model predicts.
    """
    tail_text = tok.decode(ids[0, -40:].tolist())
    expr = find_arithmetic_expression(tail_text)
    if expr is None:
        return ids, None
    real_value = evaluate_arithmetic(expr)
    if real_value is None:
        return ids, None  # detected an "expr =" shape the evaluator still won't trust -- leave it to the model

    value_str = str(int(real_value)) if float(real_value).is_integer() else str(round(real_value, 6))
    splice_ids = tok.encode(f" {value_str}")
    generated.extend(splice_ids)
    new_ids = torch.cat([ids, torch.tensor([splice_ids])], dim=1)
    splice_verified = value_str in tok.decode(splice_ids).strip()
    return new_ids, {"expr": expr, "real_value": real_value, "splice_verified": splice_verified}


def generate_with_calculator(model, tok, prompt: str, max_new_tokens: int,
                              ngram_index: set = None, ngram_n: int = 4, no_repeat_ngram_size: int = 4):
    """Same token-by-token loop as generate_with_grounding (predict_next,
    SessionTrie, no-repeat-ngram blocking -- no duplicated logic), but with
    one addition: every time generation completes a literal arithmetic
    expression right before an '=' (grounding.find_arithmetic_expression),
    the model's own digit prediction for the result is skipped entirely --
    the real value is computed (grounding.evaluate_arithmetic) and its
    tokens are spliced in directly, bypassing predict_next for that span.
    The neural net keeps mimicking the reasoning *shape* (which operation,
    in what order); it never gets to guess the digits of a result.

    Each splice records model_would_have_generated (a peek predict_next
    call at the same position, never used to make any decision -- purely
    for an honest before/after comparison) and splice_verified: whether
    re-decoding the spliced tokens actually reads back as the intended
    value. Named risk (not assumed away): encoding a value string out of
    context can hit a BPE token-boundary mismatch versus how the model
    would tokenize the same text with full left-context -- this is the
    real check for that, not a guarantee.
    """
    ids = torch.tensor([tok.encode(prompt)], dtype=torch.long)
    generated = []
    splices = []
    session_memory = SessionTrie()

    # Real gap found empirically (bench_arithmetic.py): if *prompt* itself
    # already ends in "expr =" (e.g. a caller directly asks "47 * 89 ="),
    # waiting for the in-loop check below can miss it entirely -- BPE
    # tokenization can fuse "=" together with the first digit of whatever
    # the model generates next into a single token/step, so the
    # intervention point (the moment right after "=" alone) is skipped
    # over rather than landed on. One check against the raw prompt before
    # any generation happens catches this case directly.
    ids, splice = _maybe_splice_arithmetic(tok, ids, generated)
    if splice is not None:
        splice["model_would_have_generated"] = None  # nothing generated yet at this point to compare against
        splices.append(splice)

    while len(generated) < max_new_tokens:
        next_id, info = predict_next(model, ids, ngram_index=ngram_index, ngram_n=ngram_n,
                                      session_memory=session_memory, no_repeat_ngram_size=no_repeat_ngram_size)
        generated.append(next_id)
        ids = torch.cat([ids, torch.tensor([[next_id]])], dim=1)

        new_ids, splice = _maybe_splice_arithmetic(tok, ids, generated)
        if splice is not None:
            # Peek at what the model would have predicted here instead --
            # never used to make any decision, purely recorded for an
            # honest before/after comparison. Uses the pre-splice `ids`
            # (the exact position the real value's first token replaces),
            # a fresh forward pass only paid for on an actual splice event.
            peek_id, _ = predict_next(model, ids, no_repeat_ngram_size=no_repeat_ngram_size)
            splice["model_would_have_generated"] = tok.decode([peek_id])
            splices.append(splice)
        ids = new_ids

    text = tok.decode(generated) if generated else ""
    return {"prompt": prompt, "generated_text": text, "n_tokens_generated": len(generated),
            "splices": splices}


def _score_candidate(result: dict, domain: str) -> float:
    """Turns the grounding signals generate_with_grounding already computes
    into a single scalar so candidates can be ranked. Syntax validity is a
    hard binary fact about code (either it parses or it doesn't) -- worth
    more than any amount of self-critique confidence, so it's a decisive
    bonus rather than an averaged-in term.
    """
    score = result["self_critique_score"]
    if domain == "code" and result.get("syntax_valid"):
        score += 1.0
    return score


def generate_with_resampling(model, tok, prompt: str, max_new_tokens: int, domain: str,
                              symbol_table: set = None, ngram_index: set = None, ngram_n: int = 4,
                              n_candidates: int = 5, temperature: float = 0.8, top_p: float = 1.0,
                              repetition_penalty: float = 1.0, use_chunked_state: bool = True):
    """Reject-and-resample: generate n_candidates independent completions
    and return the one with the best self-critique (+ syntax-validity for
    code) score, instead of reporting those signals only after the fact on
    a single generation. Needs temperature > 0 -- at temperature=0
    predict_next is deterministic, so every candidate would be identical
    and there would be nothing to select between.

    use_chunked_state: see generate_with_grounding's docstring -- same
    RWKV-state-carrying fix, applied per-candidate (each is independently
    primed and rolled over; state is never shared across candidates).
    """
    candidates = []
    prompt_ids = tok.encode(prompt)
    for _ in range(n_candidates):
        ids = torch.tensor([prompt_ids], dtype=torch.long)
        prompt_len = ids.size(1)
        generated = []
        session_memory = SessionTrie()  # fresh per candidate -- each is its own independent generation
        chunk_state = None
        if use_chunked_state:
            chunk_state = ChunkedState()
            chunk_state.prime(model, prompt_ids)
        for _ in range(max_new_tokens):
            next_id, info = predict_next(model, ids, chunk_state=chunk_state, ngram_index=ngram_index,
                                          ngram_n=ngram_n, temperature=temperature, top_p=top_p,
                                          repetition_penalty=repetition_penalty, session_memory=session_memory)
            generated.append(next_id)
            ids = torch.cat([ids, torch.tensor([[next_id]])], dim=1)

        text = tok.decode(generated) if generated else ""
        result = {"prompt": prompt, "generated_text": text, "n_tokens_generated": len(generated)}
        if generated:
            result["self_critique_score"] = self_critique_score(model, ids[:, :prompt_len], generated)
            if domain == "code":
                full_text = prompt + text
                result["syntax_valid"] = verify_code_syntax(full_text)
                if symbol_table is not None:
                    result["identifier_grounded"] = identifier_grounded(full_text, symbol_table)
        candidates.append(result)

    scored = [(_score_candidate(r, domain), r) for r in candidates]
    best_score, best_result = max(scored, key=lambda sr: sr[0])
    best_result["n_candidates_tried"] = n_candidates
    best_result["all_candidate_scores"] = [
        round(s, 4) if s != float("-inf") else None for s, _ in scored
    ]
    return best_result
