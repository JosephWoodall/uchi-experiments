"""First-rung test of an alternative mechanism for Ducky's arithmetic
step-sequencer: given the operations performed so far in a reduction
trace (e.g. ["MUL"] before "SUB"), predict which operation comes next.
A genuinely sequential/ordered question -- real structure exists (real
operator precedence always groups all */  before all +-, regardless of
the original random draw order), distinct from any numeric value.

Compares uchi's ported UniversalPredictor (sequence_predictor.py) against
a tiny freshly-trained Ducky model (TinyGPT) on the identical task and
data split. UniversalPredictor never touches or produces a numeric value
here -- only the categorical "which operation is next" -- honoring both
the user's explicit scoping and uchi's own numeric_plausibility.py
precedent (this predictor class is the wrong tool for numeric-value
judgments; operation-order prediction is a different, sequential
question it's actually suited for).

Usage: python3 src/eval_step_sequencer.py
"""
import json
import random
from pathlib import Path

import torch

from model import GPTConfig, TinyGPT
from sequence_predictor import UniversalPredictor
from train import compute_lm_loss

ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = ROOT / "data" / "arithmetic" / "op_sequences.json"

VOCAB = {"ADD": 0, "SUB": 1, "MUL": 2, "DIV": 3}
PAD_ID = 4
BLOCK_SIZE = 4  # max op_sequence length (n_operands in {2..5} -> 1..4 operators)
STEPS = 400
BATCH_SIZE = 32


def load_split():
    examples = json.loads(DATA_PATH.read_text())
    sequences = [e["op_sequence"] for e in examples]
    split = int(len(sequences) * 0.9)  # same 90/10 split generate_arithmetic_corpus.py used
    return sequences[:split], sequences[split:]


def eval_universal_predictor(train_seqs: list, val_seqs: list) -> dict:
    predictor = UniversalPredictor(context_length=8)
    for seq in train_seqs:
        predictor.history = []  # fresh context per sequence, trie/credibilities persist across sequences
        predictor.train(seq)

    n_correct, n_total = 0, 0
    for seq in val_seqs:
        predictor.history = []
        for i, op in enumerate(seq):
            if i > 0:  # position 0 has no real prior context to predict from
                pred, _ = predictor.predict()
                n_total += 1
                n_correct += int(pred == op)
            predictor.observe(op)
            predictor.feedback(op)
    return {"n_correct": n_correct, "n_total": n_total,
            "accuracy": n_correct / n_total if n_total else 0.0}


def encode_seq(seq: list) -> list:
    ids = [VOCAB[o] for o in seq]
    return ids + [PAD_ID] * (BLOCK_SIZE - len(ids))


def make_targets(seq: list) -> list:
    return [VOCAB[seq[t + 1]] if t + 1 < len(seq) else -1 for t in range(BLOCK_SIZE)]


def make_batch(sequences: list, batch_size: int, rng: random.Random):
    x_rows, t_rows = [], []
    for _ in range(batch_size):
        seq = rng.choice(sequences)
        x_rows.append(encode_seq(seq))
        t_rows.append(make_targets(seq))
    x = torch.tensor(x_rows, dtype=torch.long)
    targets = torch.tensor(t_rows, dtype=torch.long).unsqueeze(-1)
    return x, targets


def train_tiny_ducky(train_seqs: list, seed: int = 0) -> TinyGPT:
    torch.manual_seed(seed)
    cfg = GPTConfig(vocab_size=5, block_size=BLOCK_SIZE, d_model=16, n_layer=2, n_head=2)
    model = TinyGPT(cfg)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-3)
    rng = random.Random(seed)
    for _ in range(STEPS):
        model.train()
        opt.zero_grad()
        x, targets = make_batch(train_seqs, BATCH_SIZE, rng)
        loss = compute_lm_loss(model, x, targets, pad_id=PAD_ID)
        loss.backward()
        opt.step()
    model.eval()
    return model


@torch.no_grad()
def eval_tiny_ducky(model: TinyGPT, val_seqs: list) -> dict:
    n_correct, n_total = 0, 0
    for seq in val_seqs:
        for i in range(1, len(seq)):
            prefix = [VOCAB[o] for o in seq[:i]]
            x = torch.tensor([prefix], dtype=torch.long)
            logits, _, _, _ = model(x)
            pred_id = logits[0, -1].argmax().item()
            n_total += 1
            n_correct += int(pred_id == VOCAB[seq[i]])
    return {"n_correct": n_correct, "n_total": n_total,
            "accuracy": n_correct / n_total if n_total else 0.0}


def majority_class_baseline(train_seqs: list, val_seqs: list) -> dict:
    """Trivial "always guess the most common op" baseline -- essential
    context here, not decoration: with real operator precedence grouping
    all */  before all +-, later positions in a sequence are disproportio-
    nately ADD/SUB, so a constant prediction alone can score well above
    the naive 25% (4-way) chance rate. Neither real predictor's number
    means anything without this for comparison.
    """
    from collections import Counter
    counts = Counter()
    for seq in train_seqs:
        for op in seq[1:]:  # position 0 has no prior context, excluded exactly like the real eval loops
            counts[op] += 1
    majority_op = counts.most_common(1)[0][0]
    n_correct, n_total = 0, 0
    for seq in val_seqs:
        for i in range(1, len(seq)):
            n_total += 1
            n_correct += int(seq[i] == majority_op)
    return {"majority_op": majority_op, "n_correct": n_correct, "n_total": n_total,
            "accuracy": n_correct / n_total if n_total else 0.0}


def main():
    train_seqs, val_seqs = load_split()
    print(f"train sequences: {len(train_seqs)}, val sequences: {len(val_seqs)}")

    baseline_result = majority_class_baseline(train_seqs, val_seqs)
    print("majority-class baseline:", json.dumps(baseline_result, indent=2))

    up_result = eval_universal_predictor(train_seqs, val_seqs)
    print("UniversalPredictor:", json.dumps(up_result, indent=2))

    model = train_tiny_ducky(train_seqs)
    n_params = model.num_params()
    ducky_result = eval_tiny_ducky(model, val_seqs)
    print(f"tiny Ducky ({n_params:,} params):", json.dumps(ducky_result, indent=2))

    return {"majority_baseline": baseline_result, "universal_predictor": up_result,
            "tiny_ducky": {**ducky_result, "n_params": n_params}}


if __name__ == "__main__":
    main()
