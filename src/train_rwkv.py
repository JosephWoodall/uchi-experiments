"""Train RWKV in isolation -- base next-token objective only, no MoE/graph/
swarm/multi-token-prediction. Same tokenizer, same data, same 700-step
budget as the dense TinyGPT baseline (rj: 4.36 val loss, code: 4.81), so
the comparison is apples to apples: is RWKV viable as a backbone at all,
before testing its unique unlimited-context property separately.
"""
import argparse
import time

import torch
import torch.nn.functional as F

from data import get_lm_batch, load_lm_corpus
from rwkv_model import RWKVConfig, RWKVModel
from tokenizer import Tokenizer


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", choices=["rj", "code"], required=True)
    p.add_argument("--d-model", type=int, default=128)
    p.add_argument("--n-layer", type=int, default=4)
    p.add_argument("--block-size", type=int, default=128)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--steps", type=int, default=700)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--log-every", type=int, default=175)
    args = p.parse_args()
    torch.manual_seed(0)

    tok = Tokenizer()
    train_ids, val_ids = load_lm_corpus(args.dataset, tok)

    cfg = RWKVConfig(vocab_size=tok.vocab_size, d_model=args.d_model, n_layer=args.n_layer)
    model = RWKVModel(cfg)
    print(f"[rwkv/{args.dataset}] {model.num_params():,} params, block_size={args.block_size}")

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)

    def batch_loss(ids, bs):
        x, targets = get_lm_batch(ids, bs, args.block_size, n_future=1)
        logits, _ = model(x)
        return F.cross_entropy(logits.reshape(-1, logits.size(-1)), targets[:, :, 0].reshape(-1))

    @torch.no_grad()
    def eval_loss(n_batches=5):
        model.eval()
        losses = [batch_loss(val_ids, args.batch_size).item() for _ in range(n_batches)]
        model.train()
        return sum(losses) / len(losses)

    best_val = float("inf")
    t0 = time.time()
    for step in range(1, args.steps + 1):
        model.train()
        opt.zero_grad()
        loss = batch_loss(train_ids, args.batch_size)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()

        if step % args.log_every == 0 or step == args.steps:
            val_loss = eval_loss()
            best_val = min(best_val, val_loss)
            print(f"step {step}: wall_s={time.time()-t0:.1f} loss={loss.item():.3f} val_loss={val_loss:.3f}")

    print(f"done in {time.time()-t0:.1f}s, best val_loss={best_val:.4f}")
    torch.save(model.state_dict(), f"../runs/rwkv_{args.dataset}_best.pt")


if __name__ == "__main__":
    main()
