"""Cheapest possible test of whether an ensemble of independently-seeded
small models beats any single one -- no training needed beyond what
already exists, no new mechanism, just probability-averaging at eval
time. Directly tests the hypothesis that seed-level decorrelation (already
measured in tasks/ducky.md's tie_layers repeat-seed check) is worth
something on its own, without needing the domain specialization that
swarm.md's routing-based rejection required and never found.

Averages softmax PROBABILITIES across models (not raw logits) -- the
standard, correct way to combine independently-trained models' calibrated
confidences (logit-averaging can be dominated by whichever model happens
to produce the largest-magnitude logits, not the most calibrated one).
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


def load_model(run_name: str) -> TinyGPT:
    cfg_dict = json.loads((ROOT / "runs" / run_name / "config.json").read_text())
    size_cfg = SIZES[cfg_dict["size"]]
    cfg = GPTConfig(vocab_size=cfg_dict["vocab_size"], block_size=cfg_dict["block_size"], **size_cfg)
    model = TinyGPT(cfg)
    model.load_state_dict(torch.load(ROOT / "runs" / run_name / "model_best.pt", map_location="cpu", weights_only=True))
    model.eval()
    return model


@torch.no_grad()
def evaluate(models: list, val_ids: torch.Tensor, block_size: int, n_samples: int = 500) -> dict:
    per_model_correct = [0] * len(models)
    per_model_nll = [0.0] * len(models)
    ensemble_correct = 0
    ensemble_nll = 0.0

    for _ in range(n_samples):
        start = torch.randint(0, len(val_ids) - block_size - 1, (1,)).item()
        ctx = val_ids[start : start + block_size].unsqueeze(0)
        actual_next = val_ids[start + block_size].item()

        probs_list = []
        for i, m in enumerate(models):
            logits, _, _, _ = m(ctx)
            probs = F.softmax(logits[0, -1], dim=-1)
            probs_list.append(probs)
            pred = probs.argmax().item()
            per_model_correct[i] += int(pred == actual_next)
            per_model_nll[i] += -torch.log(probs[actual_next] + 1e-12).item()

        avg_probs = torch.stack(probs_list).mean(dim=0)
        ensemble_pred = avg_probs.argmax().item()
        ensemble_correct += int(ensemble_pred == actual_next)
        ensemble_nll += -torch.log(avg_probs[actual_next] + 1e-12).item()

    per_model_acc = [c / n_samples for c in per_model_correct]
    per_model_avg_nll = [nll / n_samples for nll in per_model_nll]

    return {
        "n_samples": n_samples,
        "per_model": [
            {"accuracy": round(a, 4), "avg_nll": round(n, 4)}
            for a, n in zip(per_model_acc, per_model_avg_nll)
        ],
        "best_single_accuracy": round(max(per_model_acc), 4),
        "avg_single_accuracy": round(sum(per_model_acc) / len(per_model_acc), 4),
        "best_single_nll": round(min(per_model_avg_nll), 4),
        "avg_single_nll": round(sum(per_model_avg_nll) / len(per_model_avg_nll), 4),
        "ensemble_accuracy": round(ensemble_correct / n_samples, 4),
        "ensemble_nll": round(ensemble_nll / n_samples, 4),
    }


def main(run_names: list, dataset: str):
    models = [load_model(name) for name in run_names]
    cfg_dicts = [json.loads((ROOT / "runs" / name / "config.json").read_text()) for name in run_names]
    vocab_size = cfg_dicts[0]["vocab_size"]
    block_size = cfg_dicts[0]["block_size"]
    tok = Tokenizer(vocab_size=vocab_size)
    _, val_ids = load_lm_corpus(dataset, tok)

    result = evaluate(models, val_ids, block_size)
    print(json.dumps(result, indent=2))

    gap = result["ensemble_accuracy"] - result["best_single_accuracy"]
    print(f"\nensemble accuracy minus BEST single model: {gap:+.4f}")
    if gap > 0.005:
        print("Real win: the ensemble beats the best individual seed, not just the average one.")
    elif gap < -0.005:
        print("Real loss: the ensemble is worse than just picking the best seed.")
    else:
        print("No meaningful difference -- ensembling isn't buying anything here.")
    return result


if __name__ == "__main__":
    run_names = sys.argv[1:-1] if len(sys.argv) > 2 else ["rj_base_s_seed57", "rj_base_s_seed58", "rj_base_s_seed59"]
    dataset = sys.argv[-1] if len(sys.argv) > 2 else "rj"
    main(run_names, dataset)
