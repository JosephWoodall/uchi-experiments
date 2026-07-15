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


def calibrate_thresholds(model, val_ids, block_size: int, fast_percentile: float = 85,
                          abstain_percentile: float = 15, n_samples: int = 500):
    """Percentile-based thresholds from this checkpoint's own confidence
    distribution: fast path = "more confident than most of what this model
    ever produces," abstain = "less confident than nearly everything this
    model produces." Returns (fast_threshold, abstain_threshold).
    """
    dist = measure_confidence_distribution(model, val_ids, block_size, n_samples)
    fast_t = torch.quantile(dist, fast_percentile / 100).item()
    abstain_t = torch.quantile(dist, abstain_percentile / 100).item()
    return fast_t, abstain_t


def predict_next(model, graph, idx: torch.Tensor, fast_threshold: float = DEFAULT_FAST_THRESHOLD,
                  abstain_threshold: float = DEFAULT_ABSTAIN_THRESHOLD, alpha: float = 0.6, beta: float = 0.4):
    """Returns (token_id or ABSTAIN, info_dict) for the next token after idx.
    fast_threshold/abstain_threshold should come from calibrate_thresholds()
    for this specific model, not the module defaults.
    """
    with torch.no_grad():
        logits, _, _, _ = model(idx[:, -model.cfg.block_size :])
        neural_logits = logits[0, -1, :]
        neural_probs = F.softmax(neural_logits, dim=-1)
        neural_conf, neural_pred = neural_probs.max(dim=-1)
        neural_conf, neural_pred = neural_conf.item(), neural_pred.item()

    if neural_conf > fast_threshold:
        return neural_pred, {"path": "fast", "confidence": round(neural_conf, 4)}

    last_token = idx[0, -1].item()
    graph_edges = graph.successors(last_token)

    if not graph_edges:
        if neural_conf < abstain_threshold:
            return ABSTAIN, {"path": "slow", "reason": "low confidence, no graph coverage",
                              "confidence": round(neural_conf, 4)}
        return neural_pred, {"path": "slow", "confidence": round(neural_conf, 4), "graph_coverage": False}

    graph_scores = torch.zeros_like(neural_logits)
    for tgt, meta in graph_edges.items():
        graph_scores[tgt] = meta["weight"] * meta["confidence"]
    combined_logits = alpha * neural_logits + beta * graph_scores
    combined_probs = F.softmax(combined_logits, dim=-1)
    combined_conf, combined_pred = combined_probs.max(dim=-1)
    combined_conf, combined_pred = combined_conf.item(), combined_pred.item()

    graph_top = max(graph_edges, key=lambda t: graph_edges[t]["weight"] * graph_edges[t]["confidence"])
    consistent = (graph_top == neural_pred) or (neural_pred in graph_edges)

    if combined_conf < abstain_threshold or not consistent:
        return ABSTAIN, {"path": "slow", "reason": "low confidence or graph/neural disagreement",
                          "neural_pred": neural_pred, "graph_top": graph_top,
                          "confidence": round(combined_conf, 4), "consistent": consistent}

    return combined_pred, {"path": "slow", "confidence": round(combined_conf, 4), "graph_coverage": True}
