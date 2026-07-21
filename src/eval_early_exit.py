"""Confidence-gated conditional compute, cheapest possible test first: does
per-layer confidence -- probed via the model's own already-trained final
norm+head applied at intermediate depth ("logit lens," nostalgebraist 2020)
-- actually identify tokens that don't need full depth, on an already-
trained checkpoint? No training, no new model, no second classifier: this
reuses runs/rj_base_m's own blocks/ln_f/head exactly as saved.

This is the load-bearing question before building a real training-time
gating mechanism (Mixture-of-Depths, Raposo et al. 2024, arXiv:2404.02258,
or CALM-style calibrated early exit, Schuster et al. 2022, arXiv:2207.07061)
-- see tasks/ducky.md's architecture-critique section (ranked idea #1) for
why this is the first thing to check, not the last.
"""
import json
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

from data import load_lm_corpus
from model import GPTConfig, TinyGPT
from tokenizer import Tokenizer
from train import SIZES

ROOT = Path(__file__).resolve().parent.parent

THRESHOLDS = [0.3, 0.5, 0.7, 0.85]


@torch.no_grad()
def layer_confidence_trace(model: TinyGPT, ctx: torch.Tensor):
    """ctx: (1, T) token ids. Returns [(pred_token, confidence), ...] at
    every depth 1..n_layer -- replicates TinyGPT.hidden_states's own block
    loop, but probes the running hidden state with the model's real
    ln_f + output projection after every block, not just at the end.
    """
    pos = torch.arange(ctx.size(1))
    tok_x = model.factored_emb.embed(ctx) if model.use_factored_embedding else model.tok_emb(ctx)
    x = tok_x + model.pos_emb(pos)
    project = model.factored_emb.project if model.use_factored_embedding else model.lm_head

    trace = []
    state = None
    for block in model.blocks:
        x, _, state = block(x, state)
        probe = model.ln_f(x)
        logits = project(probe)
        probs = F.softmax(logits[0, -1], dim=-1)
        conf, pred = probs.max(dim=-1)
        trace.append((pred.item(), conf.item()))
    return trace


@torch.no_grad()
def evaluate(model: TinyGPT, val_ids: torch.Tensor, block_size: int, n_samples: int = 500):
    n_layer = len(model.blocks)
    per_layer_conf = [[] for _ in range(n_layer)]
    per_layer_agree_final = [0] * n_layer
    per_layer_correct = [0] * n_layer
    threshold_exit_depth = {t: [] for t in THRESHOLDS}
    threshold_correct = {t: 0 for t in THRESHOLDS}

    for _ in range(n_samples):
        start = torch.randint(0, len(val_ids) - block_size - 1, (1,)).item()
        ctx = val_ids[start : start + block_size].unsqueeze(0)
        actual_next = val_ids[start + block_size].item()

        trace = layer_confidence_trace(model, ctx)
        final_pred = trace[-1][0]

        for i, (pred, conf) in enumerate(trace):
            per_layer_conf[i].append(conf)
            per_layer_agree_final[i] += int(pred == final_pred)
            per_layer_correct[i] += int(pred == actual_next)

        for t in THRESHOLDS:
            exit_depth, exit_pred = n_layer, final_pred
            for i, (pred, conf) in enumerate(trace):
                if conf >= t:
                    exit_depth, exit_pred = i + 1, pred
                    break
            threshold_exit_depth[t].append(exit_depth)
            threshold_correct[t] += int(exit_pred == actual_next)

    full_depth_accuracy = per_layer_correct[-1] / n_samples

    return {
        "n_samples": n_samples,
        "n_layer": n_layer,
        "full_depth_accuracy": round(full_depth_accuracy, 4),
        "per_layer": [
            {
                "layer": i + 1,
                "avg_confidence": round(sum(per_layer_conf[i]) / n_samples, 4),
                "agreement_with_final_layer": round(per_layer_agree_final[i] / n_samples, 4),
                "accuracy_vs_ground_truth": round(per_layer_correct[i] / n_samples, 4),
            }
            for i in range(n_layer)
        ],
        "confidence_gated_exit": [
            {
                "threshold": t,
                "avg_exit_depth": round(sum(threshold_exit_depth[t]) / n_samples, 3),
                "compute_saved_frac": round(1 - (sum(threshold_exit_depth[t]) / n_samples) / n_layer, 4),
                "accuracy": round(threshold_correct[t] / n_samples, 4),
                "accuracy_delta_vs_full_depth": round(threshold_correct[t] / n_samples - full_depth_accuracy, 4),
            }
            for t in THRESHOLDS
        ],
    }


def main(run_name: str, dataset: str):
    cfg_dict = json.loads((ROOT / "runs" / run_name / "config.json").read_text())
    # Pre-versioning checkpoints (like rj_base_m) never recorded vocab_size --
    # same fallback ducky.py uses for the same reason (see tasks/ducky.md).
    vocab_size = cfg_dict.get("vocab_size", 1024)
    tok = Tokenizer(vocab_size=vocab_size)
    size_cfg = SIZES[cfg_dict["size"]]
    cfg = GPTConfig(
        vocab_size=vocab_size,
        block_size=cfg_dict["block_size"],
        use_rwkv_hybrid=cfg_dict.get("rwkv_hybrid", False),
        attention_layers=tuple(cfg_dict.get("attention_layers", [])),
        use_bitlinear=cfg_dict.get("use_bitlinear", False),
        embedding_rank=cfg_dict.get("embedding_rank", 0),
        **size_cfg,
    )
    model = TinyGPT(cfg)
    model.load_state_dict(torch.load(ROOT / "runs" / run_name / "model_best.pt", map_location="cpu", weights_only=True))
    model.eval()

    _, val_ids = load_lm_corpus(dataset, tok)
    result = evaluate(model, val_ids, cfg_dict["block_size"])
    print(json.dumps(result, indent=2))
    return result


if __name__ == "__main__":
    run_name = sys.argv[1] if len(sys.argv) > 1 else "rj_base_m"
    dataset = sys.argv[2] if len(sys.argv) > 2 else "rj"
    main(run_name, dataset)
