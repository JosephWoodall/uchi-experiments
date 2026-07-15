"""Fast, CPU-first training loop for the ablation harness. One run = one
(dataset, arm, size) point. Every run prints a generated sample partway
through and at the end, so you can eyeball quality, not just read a loss
number -- that eyeballing is as much the point as the scaling curve.

Usage:
  python3 src/train.py --dataset rj   --arm base     --size xs
  python3 src/train.py --dataset code --arm mtp      --size s
  python3 src/train.py --dataset code --arm jepa-aux --size m
"""
import argparse
import json
import time
from pathlib import Path

import torch
import torch.nn.functional as F

from data import get_lm_batch, load_code_pairs, load_lm_corpus
from model import GPTConfig, TinyGPT
from tokenizer import Tokenizer

ROOT = Path(__file__).resolve().parent.parent
RUNS_DIR = ROOT / "runs"

# Four sizes spanning ~50K -> ~5M params at d_model/n_layer/n_head below.
# n_head chosen to divide d_model; exact counts printed at startup.
SIZES = {
    "xs": dict(d_model=32, n_layer=2, n_head=2),
    "s": dict(d_model=64, n_layer=3, n_head=4),
    "m": dict(d_model=128, n_layer=4, n_head=4),
    "l": dict(d_model=256, n_layer=6, n_head=8),
}

PROMPTS = {
    "rj": "ROMEO:",
    "code": "def ",
}


def compute_lm_loss(model, x, targets, pad_id):
    """targets: (B, T, n_future); k=0 is the standard next-token target."""
    logits, extra_logits = model(x)
    all_logits = [logits] + (extra_logits or [])
    total, count = 0.0, 0
    for k, lg in enumerate(all_logits):
        tgt = targets[:, :, k]
        valid = tgt != -1
        if not valid.any():
            continue
        loss_k = F.cross_entropy(
            lg.reshape(-1, lg.size(-1)), tgt.reshape(-1), ignore_index=-1
        )
        total = total + loss_k
        count += 1
    return total / max(count, 1)


def info_nce(a: torch.Tensor, b: torch.Tensor, temperature: float = 0.1) -> torch.Tensor:
    """Symmetric in-batch contrastive alignment (CLIP-style) between two
    views of the same examples -- no EMA target encoder needed, collapse
    is discouraged by the in-batch negatives instead.
    """
    a = F.normalize(a, dim=-1)
    b = F.normalize(b, dim=-1)
    logits = a @ b.T / temperature
    labels = torch.arange(a.size(0), device=a.device)
    return (F.cross_entropy(logits, labels) + F.cross_entropy(logits.T, labels)) / 2


def run(args):
    tok = Tokenizer()
    size_cfg = SIZES[args.size]
    cfg = GPTConfig(
        vocab_size=tok.vocab_size,
        block_size=args.block_size,
        n_future=args.n_future if args.arm == "mtp" else 1,
        proj_dim=64 if args.arm == "jepa-aux" else 0,
        **size_cfg,
    )
    model = TinyGPT(cfg)
    n_params = model.num_params()
    print(f"[{args.dataset}/{args.arm}/{args.size}] {n_params:,} params, block_size={args.block_size}")

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)

    run_name = f"{args.dataset}_{args.arm}_{args.size}"
    run_dir = RUNS_DIR / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    metrics_log = []
    samples_log = []

    pad_id = tok.sp.pad_id()
    prompt_ids = torch.tensor([tok.encode(PROMPTS[args.dataset])], dtype=torch.long)

    if args.arm == "jepa-aux":
        train_pairs, val_pairs = load_code_pairs(tok, args.block_size)

        def get_batch(pairs, bs):
            idxs = torch.randint(0, len(pairs), (bs,))
            docs = torch.stack([pairs[i][0] for i in idxs])
            codes = torch.stack([pairs[i][1] for i in idxs])
            return docs, codes

        def step_loss(pairs, bs):
            docs, codes = get_batch(pairs, bs)
            # code side: still a next-token predictor (targets shifted by 1, pad ignored)
            code_targets = torch.full((bs, args.block_size, 1), -1, dtype=torch.long)
            code_targets[:, :-1, 0] = codes[:, 1:]
            code_targets[codes == pad_id] = -1
            lm_loss = compute_lm_loss(model, codes, code_targets, pad_id)

            doc_targets = torch.full((bs, args.block_size, 1), -1, dtype=torch.long)
            doc_targets[:, :-1, 0] = docs[:, 1:]
            doc_targets[docs == pad_id] = -1
            doc_lm_loss = compute_lm_loss(model, docs, doc_targets, pad_id)

            doc_emb = model.pooled_embedding(docs, pad_id)
            code_emb = model.pooled_embedding(codes, pad_id)
            align_loss = info_nce(doc_emb, code_emb)

            return lm_loss + doc_lm_loss + args.align_weight * align_loss, {
                "code_lm_loss": lm_loss.item(),
                "doc_lm_loss": doc_lm_loss.item(),
                "align_loss": align_loss.item(),
            }

    else:
        train_ids, val_ids = load_lm_corpus(args.dataset, tok)
        n_future = args.n_future if args.arm == "mtp" else 1

        def step_loss(ids, bs):
            x, targets = get_lm_batch(ids, bs, args.block_size, n_future)
            loss = compute_lm_loss(model, x, targets, pad_id)
            return loss, {"loss": loss.item()}

    t0 = time.time()
    for step in range(1, args.steps + 1):
        model.train()
        opt.zero_grad()
        if args.arm == "jepa-aux":
            loss, extra = step_loss(train_pairs, args.batch_size)
        else:
            loss, extra = step_loss(train_ids, args.batch_size)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()

        if step % args.log_every == 0 or step == args.steps:
            entry = {"step": step, "wall_s": round(time.time() - t0, 2), **extra}
            metrics_log.append(entry)
            print(entry)

        if step % args.sample_every == 0 or step == args.steps:
            out = model.generate(prompt_ids.clone(), max_new_tokens=60)
            text = tok.decode(out[0].tolist())
            samples_log.append({"step": step, "sample": text})
            print(f"  sample @ {step}: {text!r}")

    (run_dir / "metrics.json").write_text(json.dumps(metrics_log, indent=2))
    (run_dir / "samples.json").write_text(json.dumps(samples_log, indent=2))
    torch.save(model.state_dict(), run_dir / "model.pt")
    (run_dir / "config.json").write_text(json.dumps({**vars(args), "n_params": n_params}, indent=2))
    total_wall = time.time() - t0
    print(f"done in {total_wall:.1f}s -> {run_dir}")

    return {
        "dataset": args.dataset,
        "arm": args.arm,
        "size": args.size,
        "n_params": n_params,
        "wall_s": round(total_wall, 2),
        **metrics_log[-1],
    }


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", choices=["rj", "code"], required=True)
    p.add_argument("--arm", choices=["base", "mtp", "jepa-aux"], required=True)
    p.add_argument("--size", choices=list(SIZES), default="xs")
    p.add_argument("--block-size", type=int, default=128)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--steps", type=int, default=300)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--n-future", type=int, default=2, help="mtp arm only")
    p.add_argument("--align-weight", type=float, default=0.5, help="jepa-aux arm only")
    p.add_argument("--log-every", type=int, default=25)
    p.add_argument("--sample-every", type=int, default=100)
    args = p.parse_args()
    torch.manual_seed(0)
    run(args)
