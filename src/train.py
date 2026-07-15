"""Fast, CPU-first training loop for the ablation harness. One run = one
(dataset, arm, size) point. Every run prints a generated sample partway
through and at the end, so you can eyeball quality, not just read a loss
number -- that eyeballing is as much the point as the scaling curve.

Every checkpoint also reports held-out validation loss, not just train
loss -- train loss alone is misleading for jepa-aux, whose pairs.jsonl is
only ~100 examples and gets revisited dozens of times per run.

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

from codec import N_CODES
from data import (
    AUDIO_OFFSET,
    MARK_RJ,
    MARKERS,
    PIXEL_OFFSET,
    UNIFIED_VOCAB_SIZE,
    get_joint_batch,
    get_lm_batch,
    load_code_pairs,
    load_joint_modalities,
    load_lm_corpus,
    load_modality_corpus,
)
from model import GPTConfig, TinyGPT
from tokenizer import Tokenizer

ROOT = Path(__file__).resolve().parent.parent
RUNS_DIR = ROOT / "runs"

MODALITY_DATASETS = {"pixel", "audio"}  # code-index sequences from codec.py, not BPE text
JOINT_WEIGHTS = {"rj": 0.25, "code": 0.25, "pixel": 0.25, "audio": 0.25}

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


MOE_AUX_WEIGHT = 0.01  # standard small weight for load-balancing losses


def compute_lm_loss(model, x, targets, pad_id):
    """targets: (B, T, n_future); k=0 is the standard next-token target."""
    logits, extra_logits, aux_loss = model(x)
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
    return total / max(count, 1) + MOE_AUX_WEIGHT * aux_loss


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
    is_modality = args.dataset in MODALITY_DATASETS
    is_joint = args.dataset == "joint"
    if is_modality and args.arm == "jepa-aux":
        raise ValueError("jepa-aux needs paired text views (docstring/function) -- not wired for pixel/audio")
    if is_joint and args.arm != "base":
        raise ValueError("joint dataset only wired for the base arm in this pass")

    tok = None if is_modality else Tokenizer()
    vocab_size = UNIFIED_VOCAB_SIZE if is_joint else (N_CODES if is_modality else tok.vocab_size)
    size_cfg = SIZES[args.size]
    n_future = args.n_future if args.arm == "mtp" else 1
    cfg = GPTConfig(
        vocab_size=vocab_size,
        block_size=args.block_size,
        n_future=n_future,
        proj_dim=64 if args.arm == "jepa-aux" else 0,
        moe_experts=args.moe_experts,
        moe_top_k=args.moe_top_k,
        use_bitlinear_experts=args.bitlinear_experts,
        **size_cfg,
    )
    model = TinyGPT(cfg)
    n_params = model.num_params()
    print(f"[{args.dataset}/{args.arm}/{args.size}] {n_params:,} params, block_size={args.block_size}, vocab={vocab_size}")

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)

    moe_suffix = f"_moe{args.moe_experts}" if args.moe_experts > 0 else ""
    moe_suffix += "bl" if args.bitlinear_experts else ""
    run_name = f"{args.dataset}_{args.arm}_{args.size}{moe_suffix}"
    run_dir = RUNS_DIR / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    metrics_log = []
    samples_log = []

    pad_id = 0 if is_modality else tok.sp.pad_id()

    def lm_batch_loss(ids, bs):
        x, targets = get_lm_batch(ids, bs, args.block_size, n_future)
        loss = compute_lm_loss(model, x, targets, pad_id)
        return loss, {"loss": loss.item()}

    def jepa_pair_loss(docs, codes):
        bs = docs.size(0)
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

        total = lm_loss + doc_lm_loss + args.align_weight * align_loss
        return total, {
            "code_lm_loss": lm_loss.item(),
            "doc_lm_loss": doc_lm_loss.item(),
            "align_loss": align_loss.item(),
        }

    def jepa_batch(pairs, bs):
        idxs = torch.randint(0, len(pairs), (bs,))
        docs = torch.stack([pairs[i][0] for i in idxs])
        codes = torch.stack([pairs[i][1] for i in idxs])
        return docs, codes

    if args.arm == "jepa-aux":
        train_pairs, val_pairs = load_code_pairs(tok, args.block_size)

        def train_step():
            docs, codes = jepa_batch(train_pairs, args.batch_size)
            return jepa_pair_loss(docs, codes)

        @torch.no_grad()
        def eval_step():
            model.eval()
            # val_pairs is tiny (~10) -- use all of it every time, not a sample
            docs = torch.stack([p[0] for p in val_pairs])
            codes = torch.stack([p[1] for p in val_pairs])
            _, metrics = jepa_pair_loss(docs, codes)
            model.train()
            return {f"val_{k}": v for k, v in metrics.items()}

    elif is_joint:
        joint = load_joint_modalities(tok)
        train_dict = {name: ids for name, (ids, _) in joint.items()}
        val_dict = {name: ids for name, (_, ids) in joint.items()}

        def train_step():
            x, targets = get_joint_batch(train_dict, JOINT_WEIGHTS, args.batch_size, args.block_size)
            loss = compute_lm_loss(model, x, targets, pad_id)
            return loss, {"loss": loss.item()}

        @torch.no_grad()
        def eval_step(n_batches=5):
            model.eval()
            per_modality = {name: [] for name in joint}
            for _ in range(n_batches):
                for name in joint:
                    x, targets = get_joint_batch({name: val_dict[name]}, {name: 1.0}, args.batch_size, args.block_size)
                    per_modality[name].append(compute_lm_loss(model, x, targets, pad_id).item())
            model.train()
            result = {f"val_loss_{name}": sum(v) / len(v) for name, v in per_modality.items()}
            result["val_loss"] = sum(result.values()) / len(result)
            return result

    else:
        if is_modality:
            train_ids, val_ids = load_modality_corpus(args.dataset)
        else:
            train_ids, val_ids = load_lm_corpus(args.dataset, tok)

        def train_step():
            return lm_batch_loss(train_ids, args.batch_size)

        @torch.no_grad()
        def eval_step(n_batches=5):
            model.eval()
            losses = [lm_batch_loss(val_ids, args.batch_size)[1]["loss"] for _ in range(n_batches)]
            model.train()
            return {"val_loss": sum(losses) / len(losses)}

    if is_joint:
        prompt_ids = None  # generated per-modality-marker below instead of one fixed prompt
    elif is_modality:
        prompt_ids = train_ids[:8].unsqueeze(0)
    else:
        prompt_ids = torch.tensor([tok.encode(PROMPTS[args.dataset])], dtype=torch.long)

    best_val = float("inf")
    best_step = 0
    t0 = time.time()
    for step in range(1, args.steps + 1):
        model.train()
        opt.zero_grad()
        loss, extra = train_step()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()

        if step % args.log_every == 0 or step == args.steps:
            eval_metrics = eval_step()
            entry = {"step": step, "wall_s": round(time.time() - t0, 2), **extra, **eval_metrics}
            metrics_log.append(entry)
            print(entry)

            val_key = "val_loss" if "val_loss" in eval_metrics else "val_code_lm_loss"
            if eval_metrics[val_key] < best_val:
                best_val = eval_metrics[val_key]
                best_step = step
                torch.save(model.state_dict(), run_dir / "model_best.pt")

        if step % args.sample_every == 0 or step == args.steps:
            if is_joint:
                # The interesting question here isn't fluency, it's whether
                # the model respects modality boundaries: seeded with only
                # a marker token, does it keep emitting ids in that
                # modality's range, or drift into another one's?
                names_by_marker = {v: k for k, v in MARKERS.items()}

                def describe(ids):
                    parts = []
                    for tid in ids:
                        if tid in names_by_marker:
                            parts.append(f"<{names_by_marker[tid].upper()}>")
                        elif tid < PIXEL_OFFSET:
                            parts.append(tok.decode([tid]))
                        elif tid < AUDIO_OFFSET:
                            parts.append(f"[P{tid - PIXEL_OFFSET}]")
                        else:
                            parts.append(f"[A{tid - AUDIO_OFFSET}]")
                    return " ".join(parts)

                def in_lane_frac(ids, name):
                    gen = ids[1:]  # drop the seed marker itself
                    if name in ("rj", "code"):
                        hits = sum(t < PIXEL_OFFSET for t in gen)
                    elif name == "pixel":
                        hits = sum(PIXEL_OFFSET <= t < AUDIO_OFFSET for t in gen)
                    else:
                        hits = sum(AUDIO_OFFSET <= t < MARK_RJ for t in gen)
                    return hits / len(gen)

                text_parts = []
                for name, marker in MARKERS.items():
                    seed = torch.tensor([[marker]], dtype=torch.long)
                    out = model.generate(seed, max_new_tokens=40)[0].tolist()
                    frac = in_lane_frac(out, name)
                    text_parts.append(f"{name} (in-lane {frac:.0%}): {describe(out)}")
                text = " | ".join(text_parts)
            elif is_modality:
                out = model.generate(prompt_ids.clone(), max_new_tokens=60)
                # No text rendering for code-index sequences in this chat --
                # print the raw codes (not directly visualizable/audible
                # here; codec.py's decoder can reconstruct an image/wav from
                # them if that's ever needed).
                text = str(out[0].tolist())
            else:
                out = model.generate(prompt_ids.clone(), max_new_tokens=60)
                text = tok.decode(out[0].tolist())
            samples_log.append({"step": step, "sample": text})
            print(f"  sample @ {step}: {text!r}")

    (run_dir / "metrics.json").write_text(json.dumps(metrics_log, indent=2))
    (run_dir / "samples.json").write_text(json.dumps(samples_log, indent=2))
    torch.save(model.state_dict(), run_dir / "model_final.pt")
    (run_dir / "config.json").write_text(
        json.dumps({**vars(args), "n_params": n_params, "best_step": best_step, "best_val": best_val}, indent=2)
    )
    total_wall = time.time() - t0
    print(f"done in {total_wall:.1f}s -> {run_dir} (best@{best_step}: {best_val:.4f})")

    best_entry = next((e for e in metrics_log if e["step"] == best_step), metrics_log[-1])
    return {
        "dataset": args.dataset,
        "arm": args.arm,
        "size": args.size,
        "n_params": n_params,
        "wall_s": round(total_wall, 2),
        "best_step": best_step,
        **best_entry,
    }


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", choices=["rj", "code", "pixel", "audio", "joint"], required=True)
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
    p.add_argument("--moe-experts", type=int, default=0, help="0 = dense MLP; >0 = MoE FFN with this many routed experts")
    p.add_argument("--moe-top-k", type=int, default=1)
    p.add_argument("--bitlinear-experts", action="store_true", help="quantize MoE experts (ported from uchi's BitLinear)")
    args = p.parse_args()
    torch.manual_seed(0)
    run(args)
