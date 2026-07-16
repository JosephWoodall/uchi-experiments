"""Streaming BPTT: batch_size parallel sequential streams through the
corpus, hidden state carried continuously across the ENTIRE corpus (not
reset per training example), gradients truncated every `gulp` steps --
Transformer-XL's segment-level recurrence (Dai et al. 2019, arXiv:1901.02860):
"stop gradient at the segment boundary, keep the memory."

This is a materially different regime than train_bptt.py's random-restart
K-chunk windows, not a re-run of the same idea: there, every training
example resets state to None at a random corpus position, so no training
example ever has real information before its own window boundary that
would help -- which is the likely reason three rounds of testing on that
regime (varying corpus size, step budget, and architecture) all found
KL~0.0000 at every horizon. Here, information can flow across the whole
corpus, sequentially; only the GRADIENT is truncated to `gulp` steps
(`gulp * block_size` tokens), controlling how far training can directly
credit-assign retention, not how far information can flow.

`gulp` is the hyperparameter this experiment is actually sweeping: too
small and there's negligible pressure to use carried state (approaches
train_bptt.py's own K-chunk-window behavior); too large costs more
memory/compute per step for uncertain additional benefit -- there is no
first-principles "right" answer, hence sweeping it.
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


def _detach_states(states):
    """Per-layer states -- attention layers in the hybrid carry no
    recurrent state at all (their slot is None, only RWKV layers have a
    real tuple to detach), so this must pass those through unchanged
    rather than assume every layer's slot is iterable.
    """
    if states is None:
        return None
    return [tuple(t.detach() for t in s) if s is not None else None for s in states]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", choices=["rj", "code"], required=True)
    p.add_argument("--size", default="m")
    p.add_argument("--attention-layers", type=int, nargs="*", default=[2])
    p.add_argument("--gulp", type=int, default=4, help="steps of real gradient flow before state is detached (values kept, not reset)")
    p.add_argument("--block-size", type=int, default=128)
    p.add_argument("--batch-size", type=int, default=8, help="number of parallel sequential streams")
    p.add_argument("--steps", type=int, default=2000, help="number of gulps (each gulp = `gulp` forward steps)")
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--log-every", type=int, default=200)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--num-threads", type=int, default=8)
    p.add_argument("--patience", type=int, default=0)
    p.add_argument("--min-delta", type=float, default=0.0)
    p.add_argument("--embedding-rank", type=int, default=0)
    p.add_argument("--final-chunk-only-loss", action="store_true",
                    help="score only the LAST chunk of each gulp instead of every chunk -- forces "
                         "earlier chunks to matter only by shaping a useful carried state, rather "
                         "than being separately rewarded for their own local prediction too. "
                         "Untested lever after four negative BPTT-retention rounds all varied data/"
                         "steps/regime but never the objective itself.")
    args = p.parse_args()
    torch.manual_seed(args.seed)
    torch.set_num_threads(args.num_threads)

    tok = Tokenizer()
    train_ids, val_ids = load_lm_corpus(args.dataset, tok)
    size_cfg = SIZES[args.size]
    cfg = GPTConfig(vocab_size=tok.vocab_size, block_size=args.block_size,
                     use_rwkv_hybrid=True, attention_layers=tuple(args.attention_layers),
                     embedding_rank=args.embedding_rank, **size_cfg)
    model = TinyGPT(cfg)
    suffix = "_finalonly" if args.final_chunk_only_loss else ""
    print(f"[stream-bptt/{args.dataset}] {model.num_params():,} params, gulp={args.gulp} steps "
          f"(gradient span={args.gulp * args.block_size} tokens), "
          f"batch_size={args.batch_size} parallel streams, vocab={tok.vocab_size}, "
          f"final_chunk_only_loss={args.final_chunk_only_loss}")

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)

    # Split the corpus into batch_size contiguous, non-overlapping segments,
    # one per stream, so different streams read different text -- classic
    # RNN/Transformer-XL batching, not this project's usual random-crop.
    seg_len = len(train_ids) // args.batch_size
    cursors = [i * seg_len for i in range(args.batch_size)]

    def next_gulp_chunks():
        """One gulp's worth of consecutive (x, y) chunks per stream,
        advancing each stream's cursor within its own segment and
        wrapping to that segment's start if it would run past the end.
        """
        nonlocal cursors
        chunks_x, chunks_y = [], []
        for k in range(args.gulp):
            xs, ys = [], []
            for b in range(args.batch_size):
                seg_start = b * seg_len
                seg_end = seg_start + seg_len
                if cursors[b] + args.block_size + 1 > seg_end:
                    cursors[b] = seg_start  # wrap within this stream's own segment
                c = cursors[b]
                xs.append(train_ids[c : c + args.block_size])
                ys.append(train_ids[c + 1 : c + args.block_size + 1])
                cursors[b] = c + args.block_size
            chunks_x.append(torch.stack(xs))
            chunks_y.append(torch.stack(ys))
        return chunks_x, chunks_y

    @torch.no_grad()
    def eval_loss(n_batches=5):
        # Fresh-state random crops for validation -- same convention as
        # train.py/train_bptt.py, so val numbers stay comparable to every
        # other run in this project; only the TRAINING regime changes here.
        model.eval()
        losses = []
        for _ in range(n_batches):
            idx = torch.randint(0, len(val_ids) - args.block_size - 1, (args.batch_size,))
            x = torch.stack([val_ids[i : i + args.block_size] for i in idx])
            y = torch.stack([val_ids[i + 1 : i + args.block_size + 1] for i in idx])
            logits, _, _, _ = model(x)
            losses.append(F.cross_entropy(logits.reshape(-1, logits.size(-1)), y.reshape(-1)).item())
        model.train()
        return sum(losses) / len(losses)

    best_val = float("inf")
    best_step = 0
    patience_counter = 0
    states = None
    t0 = time.time()

    for step in range(1, args.steps + 1):
        model.train()
        chunks_x, chunks_y = next_gulp_chunks()
        opt.zero_grad()
        s = states  # carried from the previous gulp, already detached
        if args.final_chunk_only_loss:
            # Earlier chunks still pass through the model (shaping s) and
            # still get real gradient via the final chunk's loss -- they
            # just aren't separately scored on their own local prediction,
            # so the only way they can reduce loss is by leaving useful
            # information in the state they hand off.
            final_loss = None
            for i, (x, y) in enumerate(zip(chunks_x, chunks_y)):
                logits, _, _, s = model(x, s)
                if i == len(chunks_x) - 1:
                    final_loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), y.reshape(-1))
            final_loss.backward()
            total_loss = final_loss
        else:
            total_loss = 0.0
            for x, y in zip(chunks_x, chunks_y):
                logits, _, _, s = model(x, s)  # gradient flows within this gulp only
                total_loss = total_loss + F.cross_entropy(logits.reshape(-1, logits.size(-1)), y.reshape(-1))
            total_loss = total_loss / args.gulp
            total_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        states = _detach_states(s)  # values persist, gradient does not

        if step % args.log_every == 0 or step == args.steps:
            val_loss = eval_loss()
            train_loss = total_loss.item()
            print(f"step {step}: wall_s={time.time()-t0:.1f} train_loss={train_loss:.3f} val_loss={val_loss:.3f}")
            if val_loss < best_val - args.min_delta:
                best_val = val_loss
                best_step = step
                patience_counter = 0
                torch.save(model.state_dict(), f"{RUNS_DIR}/streambptt_{args.dataset}_gulp{args.gulp}{suffix}_best.pt")
            else:
                patience_counter += 1
                if args.patience > 0 and patience_counter >= args.patience:
                    print(f"early stopping at step {step}: no improvement for "
                          f"{args.patience} checkpoints (best was step {best_step}: {best_val:.4f})")
                    break

    print(f"done in {time.time()-t0:.1f}s, best val_loss={best_val:.4f} at step {best_step}")
    with open(f"{RUNS_DIR}/streambptt_{args.dataset}_gulp{args.gulp}{suffix}_config.json", "w") as f:
        json.dump({"dataset": args.dataset, "size": args.size, "gulp": args.gulp,
                    "block_size": args.block_size, "batch_size": args.batch_size,
                    "attention_layers": args.attention_layers, "rwkv_hybrid": True,
                    "embedding_rank": args.embedding_rank, "best_val": best_val,
                    "best_step": best_step, "vocab_size": tok.vocab_size,
                    "final_chunk_only_loss": args.final_chunk_only_loss}, f)


if __name__ == "__main__":
    main()
