"""Part 3 of the halting/width-gating/adjacent-selective-decay round: does
restricting selective decay to just the attention-adjacent layer change
last round's verdict (uniform selective decay lost to plain RWKV on all 3
domains in the scaling sweep)? Reuses run_scaling_sweep.py's domain
loaders/config exactly (same vocab=32768/variant=balanced,
embedding_rank=32, size='s') so results are directly comparable to that
sweep's existing rwkv/selective rows -- only 2 new runs needed, not a
full re-sweep. Saves results to disk (unlike an inline one-off script)
so they survive regardless of stdout capture issues.
"""
import json
import time
from pathlib import Path

import torch

from data import ROOT as DATA_ROOT
from data import get_lm_batch
from model import GPTConfig, TinyGPT
from run_scaling_sweep import BATCH_SIZE, BLOCK_SIZE, EMBEDDING_RANK, PATIENCE, SEED, STEPS, load_domain
from tokenizer import Tokenizer
from train import SIZES, compute_lm_loss

RUNS_DIR = DATA_ROOT / "runs"
DOMAINS = ["rj", "code_core"]


def train_adjacent(domain: str, tok: Tokenizer, train_ids, val_ids) -> dict:
    torch.manual_seed(SEED)
    size_cfg = SIZES["s"]
    n_layer = size_cfg["n_layer"]
    attention_layers = (n_layer - 1,)
    adjacent = (n_layer - 2,)  # the one non-attention layer directly before the attention layer
    cfg = GPTConfig(
        vocab_size=tok.vocab_size, block_size=BLOCK_SIZE,
        use_rwkv_hybrid=True, attention_layers=attention_layers,
        selective_decay_layers=adjacent, embedding_rank=EMBEDDING_RANK,
        **size_cfg,
    )
    model = TinyGPT(cfg)
    n_params = model.num_params()
    block_stack_params = sum(p.numel() for p in model.blocks.parameters())
    opt = torch.optim.AdamW(model.parameters(), lr=3e-4)

    best_val, best_step, patience_ctr = float("inf"), 0, 0
    t0 = time.time()
    for step in range(1, STEPS + 1):
        model.train()
        opt.zero_grad()
        x, targets = get_lm_batch(train_ids, BATCH_SIZE, BLOCK_SIZE, 1)
        loss = compute_lm_loss(model, x, targets, pad_id=0)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        if step % 50 == 0 or step == STEPS:
            model.eval()
            with torch.no_grad():
                losses = [
                    compute_lm_loss(model, *get_lm_batch(val_ids, BATCH_SIZE, BLOCK_SIZE, 1), pad_id=0).item()
                    for _ in range(5)
                ]
            v = sum(losses) / len(losses)
            if v < best_val:
                best_val, best_step, patience_ctr = v, step, 0
            else:
                patience_ctr += 1
                if patience_ctr >= PATIENCE:
                    break

    wall_s = round(time.time() - t0, 2)
    run_name = f"sweep_{domain}_s_adjacent"
    run_dir = RUNS_DIR / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    result = {
        "domain": domain, "size": "s", "arch": "adjacent",
        "adjacent_layer": adjacent, "n_params": n_params,
        "block_stack_params": block_stack_params,
        "best_val": round(best_val, 4), "best_step": best_step, "wall_s": wall_s,
    }
    (run_dir / "config.json").write_text(json.dumps(result, indent=2))
    return result


def main():
    tok = Tokenizer(vocab_size=32768, variant="balanced")
    results = []
    for domain in DOMAINS:
        train_ids, val_ids = load_domain(domain, tok)
        print(f"=== {domain}/s/adjacent ===", flush=True)
        r = train_adjacent(domain, tok, train_ids, val_ids)
        print(f"RESULT {domain}:", r, flush=True)
        results.append(r)
    print("\n=== ALL RESULTS ===")
    for r in results:
        print(r)
    return results


if __name__ == "__main__":
    import resource
    results = main()
    print(f"\npeak_rss_mb: {resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024:.1f}")
