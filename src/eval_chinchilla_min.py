"""Runs bench_ducky.py's real, executable-assert benchmark against the
Chinchilla-matched code model (runs/code_base_chinchilla_min_rwkv_rank32_
tokbalanced) -- the direct answer to "does this size cross out of zero
real capability."
"""
import json
from pathlib import Path

import torch
import torch.nn.functional as F

from bench_ducky import TASKS, run_benchmark
from inference import _block_repeat_ngrams
from model import GPTConfig, TinyGPT
from tokenizer import Tokenizer
from train import SIZES

ROOT = Path(__file__).resolve().parent.parent
RUN_NAME = "code_base_chinchilla_min_rwkv_rank32_tokbalanced"


def load_model():
    cfg_dict = json.loads((ROOT / "runs" / RUN_NAME / "config.json").read_text())
    size_cfg = SIZES.get(cfg_dict["size"])
    if size_cfg is None:
        size_cfg = dict(d_model=128, n_layer=6, n_head=4)  # ad hoc size registered at train time
    cfg = GPTConfig(
        vocab_size=cfg_dict["vocab_size"], block_size=cfg_dict["block_size"],
        use_rwkv_hybrid=cfg_dict["rwkv_hybrid"], attention_layers=tuple(cfg_dict["attention_layers"]),
        embedding_rank=cfg_dict["embedding_rank"], **size_cfg,
    )
    model = TinyGPT(cfg)
    model.load_state_dict(torch.load(ROOT / "runs" / RUN_NAME / "model_best.pt", map_location="cpu", weights_only=True))
    model.eval()
    return model, cfg_dict


@torch.no_grad()
def greedy_ask(model, tok, prompt: str, max_new_tokens: int = 48, no_repeat_ngram_size: int = 4) -> str:
    ids = tok.encode(prompt)
    block_size = model.cfg.block_size
    for _ in range(max_new_tokens):
        ctx = torch.tensor([ids[-block_size:]], dtype=torch.long)
        logits, _, _, _ = model(ctx)
        step_logits = logits[0, -1, :]
        if no_repeat_ngram_size > 0:
            step_logits = _block_repeat_ngrams(step_logits, ids, no_repeat_ngram_size)
        next_id = step_logits.argmax().item()
        ids.append(next_id)
    return tok.decode(ids[len(tok.encode(prompt)):])


def main():
    model, cfg_dict = load_model()
    tok = Tokenizer(vocab_size=cfg_dict["vocab_size"], variant=cfg_dict.get("tokenizer_variant", ""))

    result = run_benchmark(lambda p: greedy_ask(model, tok, p), TASKS)
    print(json.dumps({k: v for k, v in result.items() if k != "results"}, indent=2))
    print("\nPer-task detail:")
    for r in result["results"]:
        print(f"  {r['name']}: passed={r['passed']}" + (f" error={r.get('error')}" if not r["passed"] else ""))
        print(f"    completion: {r['completion'][:120]!r}")
    return result


if __name__ == "__main__":
    main()
