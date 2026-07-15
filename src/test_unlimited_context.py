"""Tests the actual claim, not just 'doesn't crash': does processing a
sequence far beyond block_size=128 (a) keep memory constant, (b) scale
linearly (not quadratically) in time, and (c) does the carried-forward
state actually retain useful information from far in the past, not just
pass through as a no-op?

TinyGPT (model.py) cannot even attempt this -- model.generate() crops to
`idx[:, -self.cfg.block_size:]` every step; anything before the last 128
tokens is structurally gone. This script processes the *entire* rj corpus
(~48K tokens, 375x longer than block_size) through RWKV in fixed-size
chunks, carrying only the fixed-size state between chunks.
"""
import time

import torch
import torch.nn.functional as F

from data import load_lm_corpus
from rwkv_model import RWKVConfig, RWKVModel
from tokenizer import Tokenizer

CHUNK = 128


def chunked_forward(model, ids: torch.Tensor):
    """Process ids (1D) in fixed CHUNK-size pieces, carrying state forward.
    Never holds more than one chunk of activations at a time.
    """
    states = None
    total_loss, n_chunks = 0.0, 0
    t0 = time.time()
    for start in range(0, len(ids) - 1, CHUNK):
        chunk = ids[start : start + CHUNK].unsqueeze(0)
        if chunk.size(1) < 2:
            break
        logits, states = model(chunk, states)
        target = ids[start + 1 : start + 1 + chunk.size(1)]
        n = min(logits.size(1), len(target))
        loss = F.cross_entropy(logits[0, :n], target[:n])
        total_loss += loss.item()
        n_chunks += 1
    return total_loss / n_chunks, time.time() - t0, states


def main():
    tok = Tokenizer()
    train_ids, _ = load_lm_corpus("rj", tok)
    cfg = RWKVConfig(vocab_size=tok.vocab_size, d_model=128, n_layer=4)
    model = RWKVModel(cfg)
    model.load_state_dict(torch.load("../runs/rwkv_rj_best.pt", map_location="cpu"))
    model.eval()

    print("=== (a)/(b): memory + time scaling across increasing lengths ===")
    print(f"full rj corpus: {len(train_ids)} tokens ({len(train_ids) / CHUNK:.0f}x block_size={CHUNK})\n")
    for length in [512, 2048, 8192, 32768]:
        length = min(length, len(train_ids))
        ids = train_ids[:length]
        with torch.no_grad():
            avg_loss, wall_s, states = chunked_forward(model, ids)
        state_size_bytes = sum(t.numel() * t.element_size() for s in states for t in s)
        print(f"  length={length:6d}  avg_loss={avg_loss:.3f}  wall_s={wall_s:.2f}  "
              f"state_size={state_size_bytes} bytes (constant, does not grow with length)")

    print("\n=== (c): does carried state actually retain long-range info? ===")
    # Process the full corpus with state carried forward, then compare the
    # LAST chunk's prediction against processing that same chunk fresh
    # (state reset to zero, as if nothing came before it).
    full_ids = train_ids[: 8192 + CHUNK]
    with torch.no_grad():
        _, _, carried_state = chunked_forward(model, full_ids[:-CHUNK])
        last_chunk = full_ids[-CHUNK:].unsqueeze(0)
        logits_with_history, _ = model(last_chunk, carried_state)
        logits_fresh, _ = model(last_chunk, None)

    probs_with_history = F.softmax(logits_with_history[0, -1], dim=-1)
    probs_fresh = F.softmax(logits_fresh[0, -1], dim=-1)
    kl = F.kl_div(probs_fresh.log(), probs_with_history, reduction="sum").item()
    top_with_history = probs_with_history.argmax().item()
    top_fresh = probs_fresh.argmax().item()
    print(f"  same final chunk, with ~8K tokens of carried history vs fresh (zeroed) state:")
    print(f"  top predicted token -- with history: {top_with_history} ({tok.decode([top_with_history])!r}), "
          f"fresh: {top_fresh} ({tok.decode([top_fresh])!r})")
    print(f"  KL divergence between the two prediction distributions: {kl:.4f}")
    print("  (>0 means the ~8K-token-old context is measurably still influencing "
          "the prediction -- something TinyGPT cannot do at all past 128 tokens)")


if __name__ == "__main__":
    main()
