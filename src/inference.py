"""Single-model abstention + fast/slow inference -- concepts 5 and 6 from
the swarm.md post-mortem, deliberately without any swarm/voting/multiple
experts: a single model's own confidence, plus graph consistency, is
enough to drive both abstention and the cheap/expensive path decision.

Fast path: neural confidence alone is high -> trust it, skip the graph
entirely (cheap, the common case).
Slow path: neural confidence is low -> consult the graph. If the graph
disagrees with the model's own raw top pick and confidence stays low even
after blending, abstain rather than guess. This is uncertainty-driven
abstention (a single model's own calibration signal), not
disagreement-between-experts -- swarm.md's version of this needed multiple
experts to vote; this doesn't.

Thresholds are calibrated per-checkpoint from its own validation confidence
distribution (calibrate_thresholds), not hardcoded -- a first pass borrowed
swarm.md's production-scale defaults (0.85 fast / 0.3 abstain) verbatim and
they were badly miscalibrated for a 700-step toy model, whose typical
confidence runs far lower (everything abstained, including in-domain
prompts). Percentile-based thresholds against this model's *own*
distribution fix that.
"""
import torch
import torch.nn.functional as F

from grounding import (
    check_call_arity_consistency,
    identifier_grounded,
    ngram_grounded,
    self_critique_score,
    verify_code_syntax,
)
from session_memory import SessionTrie

ABSTAIN = -1

# Fallback only, used if a caller doesn't calibrate -- prefer
# calibrate_thresholds() against the actual checkpoint being used.
DEFAULT_FAST_THRESHOLD = 0.85
DEFAULT_ABSTAIN_THRESHOLD = 0.3


@torch.no_grad()
def measure_confidence_distribution(model, val_ids, block_size: int, n_samples: int = 500):
    """Max-softmax confidence on n_samples random val-set next-token
    predictions -- the empirical baseline that thresholds should be set
    relative to, not an assumed absolute scale.
    """
    model.eval()
    ids_t = val_ids if torch.is_tensor(val_ids) else torch.tensor(val_ids)
    confidences = []
    for _ in range(n_samples):
        start = torch.randint(0, len(ids_t) - block_size - 1, (1,)).item()
        chunk = ids_t[start : start + block_size].unsqueeze(0)
        logits, _, _, _ = model(chunk)
        conf = F.softmax(logits[0, -1], dim=-1).max().item()
        confidences.append(conf)
    return torch.tensor(confidences)


@torch.no_grad()
def _measure_combined_confidence_distribution(model, graph, val_ids, block_size: int,
                                                fast_threshold: float, alpha: float, beta: float,
                                                n_samples: int = 500):
    """Combined (neural+graph-blended) confidence, conditioned on actually
    reaching the slow path (raw neural confidence below fast_threshold) and
    having graph coverage -- the two conditions under which this
    distribution is actually used. Calibrating against the *unconditional*
    combined distribution (or reusing the raw-neural one, the first-pass
    bug) mismatches the scale predict_next actually sees at decision time.
    """
    model.eval()
    ids_t = val_ids if torch.is_tensor(val_ids) else torch.tensor(val_ids)
    confidences = []
    attempts = 0
    while len(confidences) < n_samples and attempts < n_samples * 20:
        attempts += 1
        start = torch.randint(0, len(ids_t) - block_size - 1, (1,)).item()
        chunk = ids_t[start : start + block_size].unsqueeze(0)
        logits, _, _, _ = model(chunk)
        neural_logits = logits[0, -1, :]
        neural_conf = F.softmax(neural_logits, dim=-1).max().item()
        if neural_conf >= fast_threshold:
            continue  # would take the fast path, this threshold never applies
        last_token = ids_t[start + block_size - 1].item()
        graph_edges = graph.successors(last_token)
        if not graph_edges:
            continue  # different branch (no-graph-coverage), different threshold
        graph_scores = torch.zeros_like(neural_logits)
        for tgt, meta in graph_edges.items():
            graph_scores[tgt] = meta["weight"] * meta["confidence"]
        combined_logits = alpha * neural_logits + beta * graph_scores
        confidences.append(F.softmax(combined_logits, dim=-1).max().item())
    return torch.tensor(confidences) if confidences else torch.tensor([0.0])


def calibrate_thresholds(model, graph, val_ids, block_size: int, fast_percentile: float = 85,
                          abstain_percentile: float = 15, slow_abstain_percentile: float = 15,
                          alpha: float = 0.6, beta: float = 0.4, n_samples: int = 500):
    """Percentile-based thresholds from this checkpoint's own confidence
    distributions -- two distributions, not one, because the slow path's
    decision is made on a differently-scaled (graph-blended) confidence
    than the fast/slow split itself. Returns (fast_threshold,
    abstain_threshold, slow_abstain_threshold).
    """
    dist = measure_confidence_distribution(model, val_ids, block_size, n_samples)
    fast_t = torch.quantile(dist, fast_percentile / 100).item()
    abstain_t = torch.quantile(dist, abstain_percentile / 100).item()

    combined_dist = _measure_combined_confidence_distribution(
        model, graph, val_ids, block_size, fast_t, alpha, beta, n_samples
    )
    slow_abstain_t = torch.quantile(combined_dist, slow_abstain_percentile / 100).item()
    return fast_t, abstain_t, slow_abstain_t


def _choose(probs: torch.Tensor, temperature: float) -> int:
    """temperature=0 -> argmax (deterministic, the original behavior).
    temperature>0 -> multinomial sample, tempered. The abstention decision
    elsewhere always uses the deterministic argmax confidence regardless --
    only which *token* gets returned when not abstaining is affected here,
    so temperature adds output diversity without destabilizing when the
    model abstains.
    """
    if temperature <= 0:
        return probs.argmax(dim=-1).item()
    tempered = torch.softmax(torch.log(probs + 1e-12) / temperature, dim=-1)
    return torch.multinomial(tempered, num_samples=1).item()


def _apply_session_memory(session_memory: SessionTrie, context: list, chosen_token: int, info: dict) -> None:
    """session_memory is the working-memory trie (session_memory.py) -- a
    third grounding signal, distinct from the graph and from n-gram
    grounding: both of those check against the training corpus, this
    checks against what THIS generation has already produced. If the
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


def predict_next(model, graph, idx: torch.Tensor, fast_threshold: float = DEFAULT_FAST_THRESHOLD,
                  abstain_threshold: float = DEFAULT_ABSTAIN_THRESHOLD,
                  slow_abstain_threshold: float = None, alpha: float = 0.6, beta: float = 0.4,
                  ngram_index: set = None, ngram_n: int = 4, temperature: float = 0.0,
                  session_memory: SessionTrie = None):
    """Returns (token_id or ABSTAIN, info_dict) for the next token after idx.
    All three thresholds should come from calibrate_thresholds() for this
    specific (model, graph) pair, not the module defaults. slow_abstain_threshold
    defaults to abstain_threshold if not given (the original, now-fixed bug's
    behavior), but calibrate_thresholds always provides its own value.

    ngram_index (from grounding.build_ngram_index), if given, is consulted
    only in the disagreement branch: n-gram grounding is direct evidence
    this exact sequence occurred in real source, which is stronger evidence
    than the graph's single-hop consistency heuristic -- it can rescue an
    otherwise-disagreement-triggered abstention, but does not override a
    pure low-confidence abstention (that's about overall uncertainty, not
    about whether the disagreement heuristic specifically was wrong).

    session_memory (session_memory.SessionTrie), if given, is consulted
    and updated on every non-abstain return -- see _apply_session_memory.
    Observability only for now: it annotates info, it does not itself
    trigger abstention or override the chosen token, matching how
    self-critique/syntax-validity started as report-only signals before
    reject-and-resample gave them teeth.

    temperature=0 (default) is fully deterministic -- every prior use of
    this function (and every number reported this session) assumed that.
    temperature>0 samples the returned token, so repeated calls on the same
    prompt can genuinely differ -- needed for generate_with_resampling to
    have anything to pick between.
    """
    if slow_abstain_threshold is None:
        slow_abstain_threshold = abstain_threshold

    with torch.no_grad():
        logits, _, _, _ = model(idx[:, -model.cfg.block_size :])
        neural_logits = logits[0, -1, :]
        neural_probs = F.softmax(neural_logits, dim=-1)
        neural_conf = neural_probs.max(dim=-1).values.item()  # abstention always uses greedy confidence
        neural_pred = _choose(neural_probs, temperature)

    if neural_conf > fast_threshold:
        info = {"path": "fast", "confidence": round(neural_conf, 4)}
        if ngram_index is not None:
            info["ngram_grounded"] = ngram_grounded(idx[0].tolist() + [neural_pred], ngram_index, ngram_n)
        _apply_session_memory(session_memory, idx[0].tolist(), neural_pred, info)
        return neural_pred, info

    last_token = idx[0, -1].item()
    graph_edges = graph.successors(last_token)

    if not graph_edges:
        if neural_conf < abstain_threshold:
            return ABSTAIN, {"path": "slow", "reason": "low confidence, no graph coverage",
                              "confidence": round(neural_conf, 4)}
        info = {"path": "slow", "confidence": round(neural_conf, 4), "graph_coverage": False}
        if ngram_index is not None:
            info["ngram_grounded"] = ngram_grounded(idx[0].tolist() + [neural_pred], ngram_index, ngram_n)
        _apply_session_memory(session_memory, idx[0].tolist(), neural_pred, info)
        return neural_pred, info

    graph_scores = torch.zeros_like(neural_logits)
    for tgt, meta in graph_edges.items():
        graph_scores[tgt] = meta["weight"] * meta["confidence"]
    combined_logits = alpha * neural_logits + beta * graph_scores
    combined_probs = F.softmax(combined_logits, dim=-1)
    combined_conf = combined_probs.max(dim=-1).values.item()  # abstention always uses greedy confidence
    combined_pred = _choose(combined_probs, temperature)

    graph_top = max(graph_edges, key=lambda t: graph_edges[t]["weight"] * graph_edges[t]["confidence"])
    consistent = (graph_top == neural_pred) or (neural_pred in graph_edges)

    grounded = None
    if ngram_index is not None:
        grounded = ngram_grounded(idx[0].tolist() + [combined_pred], ngram_index, ngram_n)

    if combined_conf < slow_abstain_threshold or (not consistent and not grounded):
        return ABSTAIN, {"path": "slow", "reason": "low confidence or graph/neural disagreement",
                          "neural_pred": neural_pred, "graph_top": graph_top,
                          "confidence": round(combined_conf, 4), "consistent": consistent,
                          "ngram_grounded": grounded}

    info = {"path": "slow", "confidence": round(combined_conf, 4), "graph_coverage": True}
    if ngram_index is not None:
        info["ngram_grounded"] = grounded
        if not consistent and grounded:
            info["reason"] = "disagreement rescued by n-gram grounding"
    _apply_session_memory(session_memory, idx[0].tolist(), combined_pred, info)
    return combined_pred, info


def generate_with_grounding(model, tok, graph, prompt: str, max_new_tokens: int, domain: str,
                             fast_threshold: float, abstain_threshold: float, slow_abstain_threshold: float,
                             symbol_table: set = None, ngram_index: set = None, ngram_n: int = 4):
    """Sequence-level wrapper: predict_next per token (stopping early on
    ABSTAIN -- the model doesn't know what comes next, so don't guess past
    that point), then post-hoc verification on the completed span. Syntax
    validity, self-critique, and identifier grounding are inherently
    sequence/span-level checks (there's no "is this syntactically valid"
    for a single token) -- n-gram grounding is the one signal granular
    enough to fold into predict_next's own per-token decision, done there
    instead of here.
    """
    ids = torch.tensor([tok.encode(prompt)], dtype=torch.long)
    prompt_len = ids.size(1)
    generated = []
    abstained_at = None
    # Fresh per call, discarded when this returns -- session-scoped, not
    # persisted (see session_memory.py). Seeding it with the prompt's own
    # tokens would double-count the prompt as "already generated"; it
    # starts empty and only accumulates tokens this call actually produces.
    session_memory = SessionTrie()

    for _ in range(max_new_tokens):
        next_id, info = predict_next(model, graph, ids, fast_threshold, abstain_threshold,
                                      slow_abstain_threshold, ngram_index=ngram_index, ngram_n=ngram_n,
                                      session_memory=session_memory)
        if next_id == ABSTAIN:
            abstained_at = len(generated)
            break
        generated.append(next_id)
        ids = torch.cat([ids, torch.tensor([[next_id]])], dim=1)

    text = tok.decode(generated) if generated else ""
    result = {"prompt": prompt, "generated_text": text, "n_tokens_generated": len(generated),
              "abstained_at_token": abstained_at}

    if generated:
        result["self_critique_score"] = self_critique_score(model, ids[:, :prompt_len], generated)
        if domain == "code":
            full_text = prompt + text
            result["syntax_valid"] = verify_code_syntax(full_text)
            if symbol_table is not None:
                result["identifier_grounded"] = identifier_grounded(full_text, symbol_table)
            arity_check = check_call_arity_consistency(full_text)
            if arity_check["consistent"] is not None:
                result["arity_consistent"] = arity_check["consistent"]
                if arity_check["conflicts"]:
                    result["arity_conflicts"] = arity_check["conflicts"]

    return result


def _score_candidate(result: dict, domain: str) -> float:
    """Turns the grounding signals generate_with_grounding already computes
    into a single scalar so candidates can be ranked. Syntax validity is a
    hard binary fact about code (either it parses or it doesn't) -- worth
    more than any amount of self-critique confidence, so it's a decisive
    bonus rather than an averaged-in term. An empty completion (abstained
    immediately) is scored below every non-empty one, even a bad one: no
    answer at all is a valid outcome elsewhere in this system, but among
    candidates that were asked to generate, one that produced nothing loses
    to one that tried.
    """
    if result["n_tokens_generated"] == 0:
        return float("-inf")
    score = result["self_critique_score"]
    if domain == "code" and result.get("syntax_valid"):
        score += 1.0
    return score


def generate_with_resampling(model, tok, graph, prompt: str, max_new_tokens: int, domain: str,
                              fast_threshold: float, abstain_threshold: float, slow_abstain_threshold: float,
                              symbol_table: set = None, ngram_index: set = None, ngram_n: int = 4,
                              n_candidates: int = 5, temperature: float = 0.8):
    """Reject-and-resample: generate n_candidates independent completions
    and return the one with the best self-critique (+ syntax-validity for
    code) score, instead of reporting those signals only after the fact on
    a single generation. Needs temperature > 0 -- at temperature=0
    predict_next is deterministic, so every candidate would be identical
    and there would be nothing to select between.

    Threads temperature into predict_next directly rather than going
    through generate_with_grounding (which doesn't expose it), since the
    abstention logic itself is untouched by temperature -- only which
    token gets returned when the model doesn't abstain.
    """
    candidates = []
    for _ in range(n_candidates):
        ids = torch.tensor([tok.encode(prompt)], dtype=torch.long)
        prompt_len = ids.size(1)
        generated = []
        abstained_at = None
        session_memory = SessionTrie()  # fresh per candidate -- each is its own independent generation
        for _ in range(max_new_tokens):
            next_id, info = predict_next(model, graph, ids, fast_threshold, abstain_threshold,
                                          slow_abstain_threshold, ngram_index=ngram_index, ngram_n=ngram_n,
                                          temperature=temperature, session_memory=session_memory)
            if next_id == ABSTAIN:
                abstained_at = len(generated)
                break
            generated.append(next_id)
            ids = torch.cat([ids, torch.tensor([[next_id]])], dim=1)

        text = tok.decode(generated) if generated else ""
        result = {"prompt": prompt, "generated_text": text, "n_tokens_generated": len(generated),
                  "abstained_at_token": abstained_at}
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
