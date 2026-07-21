"""Synthetic associative-recall task, testing whether RWKV's own recurrent
state does real long-range retention work, at toy scale, with data
specifically engineered to REQUIRE it -- per the standing rule (see
memory: no_scaleup_without_proof) to prove cheaply with representative
data before assuming architecture can't work or that scale is needed.

Five prior BPTT tests on real corpora (rj, code -- see tasks/ducky.md
Phase M) all came back negative (KL~=0, no measurable retention),
diagnosed as natural text/code rarely *forcing* long-range dependency --
there's no training pressure to develop retention when local context
already suffices. This is the standard synthetic diagnostic the SSM
literature itself uses before scale claims (associative recall /
induction heads, e.g. Gu & Dao 2023's own Mamba diagnostics) -- well-
precedented, not improvised.

Custom small integer vocabulary, no BPE tokenizer (same direct-integer
approach eval_step_sequencer.py already used): filler = 0..17 (18
symbols), KEY = 18, VALUE = 19..28 (10 symbols), QUERY = 29.
vocab_size = 30. Sequence: [filler]*L, KEY, VALUE, [filler]*L, QUERY ->
predict VALUE at the position right after QUERY. Two conditions (short L,
long L) run per architecture (pure RWKV vs hybrid) -- the real test is
whether accuracy holds as L grows, not just whether it works at short
range.

Usage: python3 src/generate_recall_corpus.py
"""
import json
import random
import time
from pathlib import Path

import torch

from model import GPTConfig, TinyGPT
from train import compute_lm_loss

ROOT = Path(__file__).resolve().parent.parent
RUN_DIR = ROOT / "runs"

FILLER_VOCAB = list(range(18))
KEY = 18
VALUE_VOCAB = list(range(19, 29))
QUERY = 29
VOCAB_SIZE = 30

STEPS = 1500
BATCH_SIZE = 32
LR = 3e-3
LOG_EVERY = 250
N_EVAL = 200


def make_sequence(rng: random.Random, L: int) -> tuple:
    """Returns (full_sequence, value) -- full_sequence's last token IS
    value (the correct answer to predict right after QUERY, the second-
    to-last token)."""
    value = rng.choice(VALUE_VOCAB)
    filler_a = [rng.choice(FILLER_VOCAB) for _ in range(L)]
    filler_b = [rng.choice(FILLER_VOCAB) for _ in range(L)]
    seq = filler_a + [KEY, value] + filler_b + [QUERY, value]
    return seq, value


def make_batch(rng: random.Random, batch_size: int, L: int):
    """x = seq[:-1], targets = seq[1:] (standard next-token shift, no
    padding needed -- every sequence in one condition has identical
    length 2L+4, since L is fixed per training run/condition). The
    position of interest (right after QUERY) is always the LAST position
    in x/targets by construction -- logits[:, -1, :] after a forward pass.
    """
    x_rows, t_rows = [], []
    for _ in range(batch_size):
        seq, _ = make_sequence(rng, L)
        x_rows.append(seq[:-1])
        t_rows.append(seq[1:])
    x = torch.tensor(x_rows, dtype=torch.long)
    targets = torch.tensor(t_rows, dtype=torch.long).unsqueeze(-1)
    return x, targets


def train_recall_model(L: int, use_rwkv_hybrid: bool, attention_layers: tuple, seed: int = 0) -> TinyGPT:
    torch.manual_seed(seed)
    block_size = 2 * L + 3  # len(seq) - 1, exact fit, no padding
    cfg = GPTConfig(vocab_size=VOCAB_SIZE, block_size=block_size, d_model=32, n_layer=3, n_head=4,
                     use_rwkv_hybrid=use_rwkv_hybrid, attention_layers=attention_layers)
    model = TinyGPT(cfg)
    opt = torch.optim.AdamW(model.parameters(), lr=LR)
    rng = random.Random(seed)
    for _ in range(STEPS):
        model.train()
        opt.zero_grad()
        x, targets = make_batch(rng, BATCH_SIZE, L)
        loss = compute_lm_loss(model, x, targets, pad_id=0)
        loss.backward()
        opt.step()
    model.eval()
    return model


@torch.no_grad()
def eval_recall_accuracy(model: TinyGPT, L: int, n_eval: int = N_EVAL, seed: int = 999) -> float:
    """Fresh, held-out (random, seed distinct from training) sequences --
    the real test: does the model correctly predict VALUE at the position
    right after QUERY."""
    rng = random.Random(seed)
    n_correct = 0
    for _ in range(n_eval):
        seq, value = make_sequence(rng, L)
        x = torch.tensor([seq[:-1]], dtype=torch.long)
        logits, _, _, _ = model(x)
        pred = logits[0, -1].argmax().item()
        n_correct += int(pred == value)
    return n_correct / n_eval


def main():
    conditions = [
        ("pure_rwkv", True, ()),
        ("hybrid", True, (2,)),
    ]
    gaps = [("short", 10), ("long", 100)]

    results = []
    for arch_name, use_rwkv_hybrid, attention_layers in conditions:
        for gap_name, L in gaps:
            t0 = time.time()
            model = train_recall_model(L, use_rwkv_hybrid, attention_layers)
            wall_s = round(time.time() - t0, 2)
            n_params = model.num_params()
            acc = eval_recall_accuracy(model, L)
            row = {"architecture": arch_name, "gap": gap_name, "L": L,
                   "n_params": n_params, "wall_s": wall_s, "recall_accuracy": acc}
            print(row)
            results.append(row)

    # Real control, run because the 4-condition sweep above was suspiciously
    # flat at chance for EVERY condition including short-gap: does the
    # copy/lookup mechanism work AT ALL when there's no gap (L=0, KEY and
    # QUERY adjacent)? If even that fails, the setup itself is broken, not
    # a real finding about retention. It doesn't fail -- L=0 reaches 100%
    # accuracy immediately, ruling out a setup bug and confirming the L=10/
    # L=100 chance-level results are real. That raised the natural
    # follow-up: exactly where does it break? Characterized directly
    # (hybrid architecture, representative -- the 4-condition sweep already
    # showed pure_rwkv and hybrid fail identically at L=10/100, so the
    # cliff's location isn't architecture-specific to re-verify per-arch).
    print("\n=== cliff characterization (hybrid, d_model=32) ===")
    cliff_results = []
    for L in [0, 1, 2, 3, 5, 10]:
        block_size = 2 * L + 3
        model = train_recall_model(L, use_rwkv_hybrid=True, attention_layers=(2,))
        acc = eval_recall_accuracy(model, L, n_eval=200)
        row = {"L": L, "recall_accuracy": acc}
        print(row)
        cliff_results.append(row)

    (RUN_DIR / "recall_sweep_results.json").write_text(
        json.dumps({"gap_sweep": results, "cliff_characterization": cliff_results}, indent=2)
    )
    print(f"\nsaved -> {RUN_DIR / 'recall_sweep_results.json'}")
    return results, cliff_results


if __name__ == "__main__":
    import resource
    main()
    print(f"peak_rss_mb: {resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024:.1f}")
