"""Cheap OOD hallucination proxy, not a research-grade benchmark: compare a
checkpoint's average generation confidence (max softmax prob per generated
token, Hendrycks & Gimpel 2017's baseline OOD signal) on in-domain vs.
out-of-domain prompts.

confidence_gap = id_confidence - ood_confidence
  positive  -> model is less sure once it's off its training distribution
               (the desired direction -- it "knows what it doesn't know")
  near zero
  or negative -> model extrapolates into OOD territory just as confidently
               as it recites training data -- confident hallucination

This exists so any arm/architecture (base, mtp, jepa-aux, moe, ...) can be
compared on this axis as cheaply as the scaling-law loss numbers, not as a
one-off manual check.

Usage: python3 src/hallucination_probe.py --run rj_base_m
"""
import argparse
import json
from pathlib import Path

import torch
import torch.nn.functional as F

from model import GPTConfig, TinyGPT
from tokenizer import Tokenizer
from train import SIZES

ROOT = Path(__file__).resolve().parent.parent
RUNS_DIR = ROOT / "runs"

ID_PROMPTS = {
    "rj": ["ROMEO:", "JULIET:", "O, "],
    "code": ["def ", "import ", "class "],
}

# Deliberately absent from both corpora (Shakespeare / Python stdlib) --
# general-knowledge claims a model that only ever saw those two sources has
# no business being confident about.
OOD_PROMPTS = [
    "The capital of France is",
    "The chemical symbol for gold is",
    "2 + 2 =",
    "The speed of light is approximately",
    "The largest planet in the solar system is",
]


def load_model(run_dir: Path, tok: Tokenizer, checkpoint: str = "model_best.pt"):
    cfg_dict = json.loads((run_dir / "config.json").read_text())
    size_cfg = SIZES[cfg_dict["size"]]
    n_future = cfg_dict.get("n_future", 1) if cfg_dict["arm"] == "mtp" else 1
    cfg = GPTConfig(
        vocab_size=tok.vocab_size,
        block_size=cfg_dict["block_size"],
        n_future=n_future,
        proj_dim=64 if cfg_dict["arm"] == "jepa-aux" else 0,
        moe_experts=cfg_dict.get("moe_experts", 0),
        moe_top_k=cfg_dict.get("moe_top_k", 1),
        use_bitlinear_experts=cfg_dict.get("bitlinear_experts", False),
        use_rwkv_hybrid=cfg_dict.get("rwkv_hybrid", False),
        attention_layers=tuple(cfg_dict.get("attention_layers", [])),
        use_bitlinear=cfg_dict.get("use_bitlinear", False),
        embedding_rank=cfg_dict.get("embedding_rank", 0),
        **size_cfg,
    )
    model = TinyGPT(cfg)
    state = torch.load(run_dir / checkpoint, map_location="cpu")
    model.load_state_dict(state)
    model.eval()
    return model, cfg_dict


@torch.no_grad()
def avg_confidence(model: TinyGPT, tok: Tokenizer, prompt: str, max_new_tokens: int = 30) -> float:
    ids = torch.tensor([tok.encode(prompt)], dtype=torch.long)
    confidences = []
    for _ in range(max_new_tokens):
        idx_cond = ids[:, -model.cfg.block_size :]
        logits, _, _, _ = model(idx_cond)
        probs = F.softmax(logits[:, -1, :], dim=-1)
        conf, next_id = probs.max(dim=-1)
        confidences.append(conf.item())
        ids = torch.cat([ids, next_id.unsqueeze(0)], dim=1)
    return sum(confidences) / len(confidences)


def run_probe(run_dir: Path, checkpoint: str = "model_best.pt") -> dict:
    tok = Tokenizer()
    model, cfg_dict = load_model(run_dir, tok, checkpoint)
    dataset = cfg_dict["dataset"]

    id_confs = [avg_confidence(model, tok, p) for p in ID_PROMPTS.get(dataset, ["The"])]
    ood_confs = [avg_confidence(model, tok, p) for p in OOD_PROMPTS]

    id_mean = sum(id_confs) / len(id_confs)
    ood_mean = sum(ood_confs) / len(ood_confs)

    result = {
        "run": run_dir.name,
        "checkpoint": checkpoint,
        "id_confidence": round(id_mean, 4),
        "ood_confidence": round(ood_mean, 4),
        "confidence_gap": round(id_mean - ood_mean, 4),
    }
    print(result)
    return result


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--run", required=True, help="run directory name under runs/, e.g. rj_base_m")
    p.add_argument("--checkpoint", default="model_best.pt")
    args = p.parse_args()
    run_probe(RUNS_DIR / args.run, args.checkpoint)
