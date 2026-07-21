"""Trained counterpart to eval_early_exit.py's untrained logit-lens probe:
evaluates a checkpoint trained with GPTConfig(use_halting=True) (model.py's
halting_loss, train.py's --use-halting), using each block's TRAINED
halt_head sigmoid output as the exit-confidence signal instead of reusing
the (untrained-for-this-purpose) final head at intermediate depth. Same
sampling/threshold-sweep methodology as eval_early_exit.py, same THRESHOLDS,
for a direct, apples-to-apples before/after comparison -- see
tasks/ducky.md's architecture-critique section (ranked idea #1).
"""
import json
import sys
from pathlib import Path

import torch

from data import load_lm_corpus
from eval_early_exit import THRESHOLDS
from model import GPTConfig, TinyGPT
from tokenizer import Tokenizer
from train import SIZES

ROOT = Path(__file__).resolve().parent.parent


@torch.no_grad()
def layer_halt_trace(model: TinyGPT, ctx: torch.Tensor):
    """Like eval_early_exit.py's layer_confidence_trace, but the confidence
    signal is the TRAINED halt_head's sigmoid output, not the untrained
    logit-lens softmax-max -- the predicted token still comes from the same
    logit-lens projection either way, only the exit-confidence signal
    changes (matching model.py's halting_loss: only the halt decision is
    learned, not the projection it's predicting agreement with).
    """
    pos = torch.arange(ctx.size(1))
    tok_x = model.factored_emb.embed(ctx) if model.use_factored_embedding else model.tok_emb(ctx)
    x = tok_x + model.pos_emb(pos)
    project = model.factored_emb.project if model.use_factored_embedding else model.lm_head

    n_layer = len(model.blocks)
    trace = []
    state = None
    for i, block in enumerate(model.blocks):
        x, _, state = block(x, state)
        logits = project(model.ln_f(x))
        pred = logits[0, -1].argmax().item()
        if i < n_layer - 1:
            conf = torch.sigmoid(model.halt_heads[i](x[0, -1])).item()
        else:
            conf = 1.0  # final layer: nothing left to predict, always "confident enough"
        trace.append((pred, conf))
    return trace


@torch.no_grad()
def evaluate(model: TinyGPT, val_ids: torch.Tensor, block_size: int, n_samples: int = 500):
    """Identical structure/metrics to eval_early_exit.py's evaluate() --
    same n_samples, same THRESHOLDS -- so the two JSON outputs are directly
    comparable row-for-row.
    """
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

        trace = layer_halt_trace(model, ctx)
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
        use_halting=cfg_dict.get("use_halting", False),
        **size_cfg,
    )
    assert cfg.use_halting, f"{run_name} wasn't trained with --use-halting -- use eval_early_exit.py instead"
    model = TinyGPT(cfg)
    model.load_state_dict(torch.load(ROOT / "runs" / run_name / "model_best.pt", map_location="cpu", weights_only=True))
    model.eval()

    _, val_ids = load_lm_corpus(dataset, tok)
    result = evaluate(model, val_ids, cfg_dict["block_size"])
    print(json.dumps(result, indent=2))
    return result


if __name__ == "__main__":
    run_name = sys.argv[1] if len(sys.argv) > 1 else "rj_base_m_halt_seed57"
    dataset = sys.argv[2] if len(sys.argv) > 2 else "rj"
    main(run_name, dataset)
