"""One-off toy training run on the new synthetic arithmetic step-trace
domain (data/arithmetic/, generate_arithmetic_corpus.py) -- reuses
data._tokenize_corpus directly (same precedent run_scaling_sweep.py set
for one-off domains like "terminal"/"code_core") instead of touching
data.py's shared load_lm_corpus dict for a single toy run.

The model is NOT being asked to learn arithmetic -- inference.py's
generate_with_calculator supplies the real values at generation time.
This run only needs to teach the step-trace textual SHAPE ("Step N: a op
b = " ... "Answer: ") well enough that the model attempts it, so there's
something for the calculator-splice mechanism to intercept.

Usage: python3 src/train_arithmetic.py
"""
import argparse
import json
import time
from pathlib import Path

import torch

from data import _tokenize_corpus, get_lm_batch
from model import GPTConfig, TinyGPT
from tokenizer import Tokenizer
from train import SIZES, compute_lm_loss

ROOT = Path(__file__).resolve().parent.parent

STEPS = 3000
PATIENCE = 6
BATCH_SIZE = 32
BLOCK_SIZE = 128
LR = 3e-4
LOG_EVERY = 100


def main(chained: bool = False):
    torch.manual_seed(0)
    torch.set_num_threads(8)

    prefix = "chained_" if chained else ""
    run_name = "arithmetic_chained_base_s_rwkv" if chained else "arithmetic_base_s_rwkv"
    run_dir = ROOT / "runs" / run_name

    tok = Tokenizer(vocab_size=1024)
    train_text = (ROOT / "data" / "arithmetic" / f"{prefix}train.txt").read_text()
    val_text = (ROOT / "data" / "arithmetic" / f"{prefix}val.txt").read_text()
    train_ids = _tokenize_corpus(tok, f"arithmetic_{prefix}train", train_text)
    val_ids = _tokenize_corpus(tok, f"arithmetic_{prefix}val", val_text)
    print(f"train tokens: {len(train_ids)}, val tokens: {len(val_ids)}")

    size_cfg = SIZES["s"]
    cfg = GPTConfig(vocab_size=tok.vocab_size, block_size=BLOCK_SIZE,
                     use_rwkv_hybrid=True, attention_layers=(2,), **size_cfg)
    model = TinyGPT(cfg)
    n_params = model.num_params()
    print(f"[arithmetic/base/s] {n_params:,} params, vocab={tok.vocab_size}")

    opt = torch.optim.AdamW(model.parameters(), lr=LR)
    pad_id = tok.sp.pad_id()

    best_val = float("inf")
    best_step = 0
    patience_counter = 0
    metrics_log = []
    t0 = time.time()

    for step in range(1, STEPS + 1):
        model.train()
        opt.zero_grad()
        x, targets = get_lm_batch(train_ids, BATCH_SIZE, BLOCK_SIZE, 1)
        loss = compute_lm_loss(model, x, targets, pad_id)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()

        if step % LOG_EVERY == 0 or step == STEPS:
            model.eval()
            with torch.no_grad():
                losses = []
                for _ in range(5):
                    vx, vt = get_lm_batch(val_ids, BATCH_SIZE, BLOCK_SIZE, 1)
                    losses.append(compute_lm_loss(model, vx, vt, pad_id).item())
                val_loss = sum(losses) / len(losses)
            entry = {"step": step, "wall_s": round(time.time() - t0, 2),
                      "loss": loss.item(), "val_loss": val_loss}
            metrics_log.append(entry)
            print(entry)

            if val_loss < best_val:
                best_val = val_loss
                best_step = step
                patience_counter = 0
                run_dir.mkdir(parents=True, exist_ok=True)
                torch.save(model.state_dict(), run_dir / "model_best.pt")
            else:
                patience_counter += 1
                if PATIENCE > 0 and patience_counter >= PATIENCE:
                    print(f"early stopping at step {step}: no improvement for "
                          f"{PATIENCE} checkpoints (best was step {best_step}: {best_val:.4f})")
                    break

    prompt = "Step 1:"
    ids = torch.tensor([tok.encode(prompt)], dtype=torch.long)
    out = model.generate(ids, max_new_tokens=60)
    sample = tok.decode(out[0].tolist())
    print(f"sample: {sample!r}")

    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "metrics.json").write_text(json.dumps(metrics_log, indent=2))
    (run_dir / "config.json").write_text(json.dumps({
        "dataset": "arithmetic_chained" if chained else "arithmetic", "arm": "base", "size": "s",
        "block_size": BLOCK_SIZE, "rwkv_hybrid": True, "attention_layers": [2], "vocab_size": tok.vocab_size,
        "n_params": n_params, "best_step": best_step, "best_val": best_val,
    }, indent=2))
    print(f"done -> {run_dir} (best@{best_step}: {best_val:.4f})")


if __name__ == "__main__":
    import resource
    p = argparse.ArgumentParser()
    p.add_argument("--chained", action="store_true", help="train on the operand-chained corpus variant instead")
    args = p.parse_args()
    main(chained=args.chained)
    print(f"peak_rss_mb: {resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024:.1f}")
