"""Cross-chunk BPTT training: unlike train.py's random-crop training (each
128-token window trained independently, RWKV state reset every call), this
processes K consecutive chunks from a contiguous span with the RWKV state
carried WITHOUT detaching across all K, so the loss on chunk K can push
gradients back into how earlier chunks' processing shaped the carried
state -- the only way to actually reward long-range retention. Confirmed
empirically why this matters: isolated-crop training gave KL=0.0 between
carried-state and fresh-state predictions after 8K tokens -- nothing in
that training ever rewarded keeping information around that long.
"""
import argparse
import json
import time

import torch
import torch.nn.functional as F

from data import load_lm_corpus
from model import GPTConfig, TinyGPT
from tokenizer import Tokenizer
from train import SIZES

RUNS_DIR = "../runs"


def get_bptt_batch(ids, batch_size, block_size, k_chunks):
    """K consecutive chunks from B random contiguous spans. Returns lists
    of K (x, y) tensor pairs, each (B, block_size), in temporal order.
    """
    span = block_size * k_chunks
    starts = torch.randint(0, len(ids) - span - 1, (batch_size,)).tolist()
    chunks_x, chunks_y = [], []
    for k in range(k_chunks):
        x = torch.stack([ids[s + k * block_size : s + (k + 1) * block_size] for s in starts])
        y = torch.stack([ids[s + k * block_size + 1 : s + (k + 1) * block_size + 1] for s in starts])
        chunks_x.append(x)
        chunks_y.append(y)
    return chunks_x, chunks_y


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", choices=["rj", "code"], required=True)
    p.add_argument("--size", default="m")
    p.add_argument("--attention-layers", type=int, nargs="*", default=[2])
    p.add_argument("--k-chunks", type=int, default=4, help="chunks per BPTT window, state carried across all K")
    p.add_argument("--block-size", type=int, default=128)
    p.add_argument("--batch-size", type=int, default=16, help="smaller than train.py's 32 -- K chunks held in the graph at once")
    p.add_argument("--steps", type=int, default=700)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--log-every", type=int, default=175)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--num-threads", type=int, default=8, help="measured optimal on this machine, see train.py")
    args = p.parse_args()
    torch.manual_seed(args.seed)
    torch.set_num_threads(args.num_threads)

    tok = Tokenizer()
    train_ids, val_ids = load_lm_corpus(args.dataset, tok)
    size_cfg = SIZES[args.size]
    cfg = GPTConfig(vocab_size=tok.vocab_size, block_size=args.block_size,
                     use_rwkv_hybrid=True, attention_layers=tuple(args.attention_layers), **size_cfg)
    model = TinyGPT(cfg)
    print(f"[bptt/{args.dataset}] {model.num_params():,} params, k_chunks={args.k_chunks}, "
          f"effective_span={args.k_chunks * args.block_size} tokens")

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)

    def bptt_loss(ids, bs):
        chunks_x, chunks_y = get_bptt_batch(ids, bs, args.block_size, args.k_chunks)
        states = None
        total_loss = 0.0
        for x, y in zip(chunks_x, chunks_y):
            logits, _, _, states = model(x, states)  # states NOT detached -- gradient flows back across chunks
            total_loss = total_loss + F.cross_entropy(logits.reshape(-1, logits.size(-1)), y.reshape(-1))
        return total_loss / args.k_chunks

    @torch.no_grad()
    def eval_loss(n_batches=5):
        model.eval()
        losses = [bptt_loss(val_ids, args.batch_size).item() for _ in range(n_batches)]
        model.train()
        return sum(losses) / len(losses)

    best_val = float("inf")
    t0 = time.time()
    for step in range(1, args.steps + 1):
        model.train()
        opt.zero_grad()
        loss = bptt_loss(train_ids, args.batch_size)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        if step % args.log_every == 0 or step == args.steps:
            val_loss = eval_loss()
            best_val = min(best_val, val_loss)
            print(f"step {step}: wall_s={time.time()-t0:.1f} loss={loss.item():.3f} val_loss={val_loss:.3f}")

    print(f"done in {time.time()-t0:.1f}s, best val_loss={best_val:.4f}")
    torch.save(model.state_dict(), f"{RUNS_DIR}/bptt_{args.dataset}_best.pt")
    with open(f"{RUNS_DIR}/bptt_{args.dataset}_config.json", "w") as f:
        json.dump({"dataset": args.dataset, "size": args.size, "k_chunks": args.k_chunks,
                    "block_size": args.block_size, "attention_layers": args.attention_layers,
                    "rwkv_hybrid": True, "best_val": best_val}, f)


if __name__ == "__main__":
    main()
