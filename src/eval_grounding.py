"""Does the graph+abstention mechanism actually improve accuracy, or just
add complexity? Standard selective-prediction check: is accuracy on
ANSWERED (non-abstained) tokens meaningfully higher than the pure neural
baseline's unconditional accuracy? If not, the mechanism isn't doing
useful selective filtering, just adding noise on top of the same model --
exactly the gap swarm.md's own postmortem flagged (a difference from
baseline isn't the same as an improvement over it), which nothing in
Ducky's grounding layer had actually closed until this check.
"""
import json
import sys

import torch

from data import load_lm_corpus
from graph import add_model_prediction_edges, build_graph
from inference import ABSTAIN, calibrate_thresholds, predict_next
from model import GPTConfig, TinyGPT
from tokenizer import Tokenizer
from train import SIZES


@torch.no_grad()
def evaluate(model, graph, val_ids, block_size, fast_t, abstain_t, slow_abstain_t, n_samples=500):
    baseline_correct = 0
    grounded_correct = 0
    grounded_answered = 0
    total = 0

    for _ in range(n_samples):
        start = torch.randint(0, len(val_ids) - block_size - 1, (1,)).item()
        ctx = val_ids[start : start + block_size].unsqueeze(0)
        actual_next = val_ids[start + block_size].item()

        logits, _, _, _ = model(ctx)
        baseline_pred = logits[0, -1].argmax().item()
        if baseline_pred == actual_next:
            baseline_correct += 1

        pred, _ = predict_next(model, graph, ctx, fast_t, abstain_t, slow_abstain_t)
        if pred != ABSTAIN:
            grounded_answered += 1
            if pred == actual_next:
                grounded_correct += 1
        total += 1

    return {
        "baseline_accuracy": round(baseline_correct / total, 4),
        "coverage": round(grounded_answered / total, 4),
        "grounded_accuracy_on_answered": round(grounded_correct / grounded_answered, 4) if grounded_answered else 0.0,
        "n": total,
    }


def main(run_name: str, dataset: str):
    tok = Tokenizer()
    cfg_dict = json.loads(open(f"../runs/{run_name}/config.json").read())
    size_cfg = SIZES[cfg_dict["size"]]
    cfg = GPTConfig(vocab_size=tok.vocab_size, block_size=cfg_dict["block_size"],
                     use_rwkv_hybrid=cfg_dict.get("rwkv_hybrid", False),
                     attention_layers=tuple(cfg_dict.get("attention_layers", [])),
                     use_bitlinear=cfg_dict.get("use_bitlinear", False),
                     embedding_rank=cfg_dict.get("embedding_rank", 0), **size_cfg)
    model = TinyGPT(cfg)
    model.load_state_dict(torch.load(f"../runs/{run_name}/model_best.pt", map_location="cpu"))
    model.eval()

    _, val_ids = load_lm_corpus(dataset, tok)
    code_source = open("../data/code/corpus.txt").read()
    rj_ids = torch.load("../data/cache/rj.pt")
    code_ids = torch.load("../data/cache/code.pt")
    graph = build_graph(tok, code_source, rj_ids, code_ids)
    add_model_prediction_edges(graph, model, val_ids, confidence_threshold=0.95, n_samples=300)

    fast_t, abstain_t, slow_abstain_t = calibrate_thresholds(model, graph, val_ids, cfg_dict["block_size"])
    print(f"calibrated: fast={fast_t:.4f} abstain={abstain_t:.4f} slow_abstain={slow_abstain_t:.4f}")

    result = evaluate(model, graph, val_ids, cfg_dict["block_size"], fast_t, abstain_t, slow_abstain_t)
    print(json.dumps(result, indent=2))

    gap = result["grounded_accuracy_on_answered"] - result["baseline_accuracy"]
    print(f"\naccuracy gap (grounded-on-answered minus baseline): {gap:+.4f}")
    if gap > 0.02:
        print("Net positive: selective filtering is finding genuinely easier/more-certain cases.")
    elif gap < -0.02:
        print("Net negative: the mechanism is answering on cases it does WORSE on than baseline -- a real problem.")
    else:
        print("No meaningful difference: grounding isn't doing useful selective filtering here, just adding abstention noise.")


if __name__ == "__main__":
    run_name = sys.argv[1] if len(sys.argv) > 1 else "rj_base_m"
    dataset = sys.argv[2] if len(sys.argv) > 2 else "rj"
    main(run_name, dataset)
