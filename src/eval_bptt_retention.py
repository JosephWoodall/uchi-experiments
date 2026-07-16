"""Does a BPTT-trained checkpoint (train_bptt.py) actually retain and use
distant context, or does carried RWKV state pass through as a no-op despite
training loss going down? Same methodology as test_unlimited_context.py's
part (c) -- compare the model's prediction on a fixed final chunk with the
real carried state (built from everything before it) against the same
chunk processed from a fresh, zeroed state -- but run across several
horizons (128 through 640 tokens) instead of one, and pointed at whichever
BPTT run directory is given, not hardcoded to one checkpoint.

KL > 0 means the distant context measurably changed the prediction --
genuine retention, not a pass-through. KL ~ 0 at every horizon (the result
every prior BPTT round produced, see tasks/ducky.md) means the carried
state isn't being used for anything a fresh state wouldn't already give.
"""
import argparse
import json

import torch
import torch.nn.functional as F

from data import load_lm_corpus
from model import GPTConfig, TinyGPT
from tokenizer import Tokenizer
from train import SIZES

CHUNK = 128
HORIZONS = [128, 256, 384, 512, 640]


def chunked_forward(model, ids: torch.Tensor):
    """Process ids in fixed CHUNK-size pieces, carrying state forward.
    Returns the final carried state (never holds more than one chunk of
    activations at a time, same as test_unlimited_context.py).
    """
    states = None
    with torch.no_grad():
        for start in range(0, len(ids) - 1, CHUNK):
            chunk = ids[start : start + CHUNK].unsqueeze(0)
            if chunk.size(1) < 2:
                break
            _, _, _, states = model(chunk, states)
    return states


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run-name", default="bptt_code", help="prefix used by train_bptt.py: <run-name>_best.pt / _config.json")
    p.add_argument("--dataset", default=None, help="defaults to the value recorded in <run-name>_config.json")
    p.add_argument("--horizons", type=int, nargs="*", default=None,
                   help="override the default 128-640 span -- streaming checkpoints can carry state "
                        "across the whole corpus, so testing only up to 640 would understate what "
                        "they were actually trained to use")
    args = p.parse_args()
    horizons = args.horizons or HORIZONS

    cfg_dict = json.loads(open(f"../runs/{args.run_name}_config.json").read())
    dataset = args.dataset or cfg_dict["dataset"]
    vocab_size = cfg_dict.get("vocab_size", 1024)  # pre-versioning runs were all vocab=1024

    tok = Tokenizer(vocab_size=vocab_size)
    train_ids, _ = load_lm_corpus(dataset, tok)

    size_cfg = SIZES[cfg_dict["size"]]
    cfg = GPTConfig(vocab_size=tok.vocab_size, block_size=cfg_dict["block_size"],
                     use_rwkv_hybrid=cfg_dict.get("rwkv_hybrid", True),
                     attention_layers=tuple(cfg_dict.get("attention_layers", [])), **size_cfg)
    model = TinyGPT(cfg)
    model.load_state_dict(torch.load(f"../runs/{args.run_name}_best.pt", map_location="cpu"))
    model.eval()

    print(f"[{args.run_name}] {model.num_params():,} params, dataset={dataset}, vocab={vocab_size}, "
          f"best_val={cfg_dict.get('best_val')} @ step {cfg_dict.get('best_step')}")
    print(f"corpus length available: {len(train_ids)} tokens\n")

    print("does carried state (built from real history) change the prediction on the "
          "same final chunk vs a fresh/zeroed state, at increasing horizons?\n")
    for horizon in horizons:
        span = horizon + CHUNK
        if span > len(train_ids):
            print(f"  horizon={horizon:4d}: skipped, corpus too short ({len(train_ids)} < {span})")
            continue
        full_ids = train_ids[:span]
        carried_state = chunked_forward(model, full_ids[:-CHUNK])
        last_chunk = full_ids[-CHUNK:].unsqueeze(0)
        with torch.no_grad():
            logits_with_history, _, _, _ = model(last_chunk, carried_state)
            logits_fresh, _, _, _ = model(last_chunk, None)

        probs_with_history = F.softmax(logits_with_history[0, -1], dim=-1)
        probs_fresh = F.softmax(logits_fresh[0, -1], dim=-1)
        kl = F.kl_div(probs_fresh.log(), probs_with_history, reduction="sum").item()
        top_hist = probs_with_history.argmax().item()
        top_fresh = probs_fresh.argmax().item()
        print(f"  horizon={horizon:4d} tokens: KL={kl:.4f}  "
              f"top_with_history={top_hist} ({tok.decode([top_hist])!r})  "
              f"top_fresh={top_fresh} ({tok.decode([top_fresh])!r})  "
              f"{'SAME token' if top_hist == top_fresh else 'DIFFERENT token'}")

    print("\n(KL > 0 at a horizon means that much distant context measurably changed "
          "the prediction -- genuine retention. KL ~ 0.0000 at every horizon means the "
          "carried state isn't being used for anything a fresh state wouldn't already give.)")


if __name__ == "__main__":
    main()
