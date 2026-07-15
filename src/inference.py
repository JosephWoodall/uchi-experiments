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
"""
import torch
import torch.nn.functional as F

FAST_CONFIDENCE_THRESHOLD = 0.85
ABSTAIN_THRESHOLD = 0.3
ABSTAIN = -1


def predict_next(model, graph, idx: torch.Tensor, alpha: float = 0.6, beta: float = 0.4):
    """Returns (token_id or ABSTAIN, info_dict) for the next token after idx."""
    with torch.no_grad():
        logits, _, _, _ = model(idx[:, -model.cfg.block_size :])
        neural_logits = logits[0, -1, :]
        neural_probs = F.softmax(neural_logits, dim=-1)
        neural_conf, neural_pred = neural_probs.max(dim=-1)
        neural_conf, neural_pred = neural_conf.item(), neural_pred.item()

    if neural_conf > FAST_CONFIDENCE_THRESHOLD:
        return neural_pred, {"path": "fast", "confidence": round(neural_conf, 4)}

    last_token = idx[0, -1].item()
    graph_edges = graph.successors(last_token)

    if not graph_edges:
        if neural_conf < ABSTAIN_THRESHOLD + 0.2:
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

    if combined_conf < ABSTAIN_THRESHOLD or not consistent:
        return ABSTAIN, {"path": "slow", "reason": "low confidence or graph/neural disagreement",
                          "neural_pred": neural_pred, "graph_top": graph_top,
                          "confidence": round(combined_conf, 4), "consistent": consistent}

    return combined_pred, {"path": "slow", "confidence": round(combined_conf, 4), "graph_coverage": True}
