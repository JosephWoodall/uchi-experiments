"""Deterministic, real-by-construction synthetic arithmetic step-trace
corpus -- same "synthetic but real, not fabricated" precedent as
data/code/synthetic_relational.txt: nothing here is invented, every
number is Python's own arithmetic.

Built for two purposes: (1) a toy training corpus so Ducky can learn to
mimic this step-trace textual SHAPE (grounding.py's calculator-splice
mechanism supplies the real values at generation time -- the model never
needs to actually learn arithmetic, just the reasoning scaffold); (2) the
per-step operation-order labels eval_step_sequencer.py compares
UniversalPredictor against, kept separate from any numeric value.

Usage: python3 src/generate_arithmetic_corpus.py
"""
import json
import random
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = ROOT / "data" / "arithmetic"

_OPS = ["+", "-", "*", "/"]
_HIGH_PREC = {"*", "/"}
_OP_NAMES = {"+": "ADD", "-": "SUB", "*": "MUL", "/": "DIV"}


def _reduce_with_steps(operands: list, ops: list):
    """Real Python operator precedence: reduce all */ pairs left-to-right
    first, then all +- pairs left-to-right, recording each individual
    binary reduction as a (a, op, b, result) step in the exact order
    Python itself would evaluate them. Returns (steps, final_result), or
    None if a division doesn't divide evenly -- rejected here and
    resampled by the caller rather than constructed analytically, since
    a clean-division constraint is otherwise awkward to satisfy when
    reduction order chains through an intermediate multiplication result.
    """
    ops_left = list(ops)
    vals = list(operands)
    steps = []

    def reduce_pass(target_ops):
        i = 0
        while i < len(ops_left):
            if ops_left[i] not in target_ops:
                i += 1
                continue
            a, b, op = vals[i], vals[i + 1], ops_left[i]
            if op == "+":
                r = a + b
            elif op == "-":
                r = a - b
            elif op == "*":
                r = a * b
            else:
                if b == 0 or a % b != 0:
                    return False
                r = a // b
            steps.append((a, op, b, r))
            vals[i:i + 2] = [r]
            ops_left[i:i + 1] = []
        return True

    if not reduce_pass(_HIGH_PREC):
        return None
    if not reduce_pass({"+", "-"}):
        return None
    return steps, vals[0]


def make_expression(rng: random.Random, n_operands: int):
    """Rejection-samples until every intermediate step (including any
    division) is clean and the final result stays in a readable range --
    simple and robust, no analytic construction needed."""
    while True:
        operands = [rng.randint(1, 20) for _ in range(n_operands)]
        ops = [rng.choice(_OPS) for _ in range(n_operands - 1)]
        result = _reduce_with_steps(operands, ops)
        if result is None:
            continue
        steps, final = result
        if abs(final) <= 10_000:
            return operands, ops, steps, final


def render_trace(steps: list, final: int) -> str:
    lines = [f"Step {i}: {a} {op} {b} = {r}" for i, (a, op, b, r) in enumerate(steps, 1)]
    lines.append(f"Answer: {final}")
    return "\n".join(lines) + "\n\n"


def op_sequence(steps: list) -> list:
    return [_OP_NAMES[s[1]] for s in steps]


def make_chained_expression(rng: random.Random, n_steps: int):
    """Unlike make_expression (all operands drawn independently up front,
    precedence-reduced after the fact), each step's FIRST operand here IS
    the previous step's real result -- a genuine operand-CHAIN, not just a
    step-shaped trace. Directly targets Part A's honestly-flagged gap: the
    calculator splice guarantees each detected expression is correct, but
    the base model was never taught to carry a real prior result forward
    as its own next operand, because the original corpus had no such
    structure to learn from at all. This one does.
    """
    while True:
        value = rng.randint(1, 20)
        steps = []
        ok = True
        for _ in range(n_steps):
            op = rng.choice(_OPS)
            b = rng.randint(1, 20)
            if op == "+":
                r = value + b
            elif op == "-":
                r = value - b
            elif op == "*":
                r = value * b
            else:
                if b == 0 or value % b != 0:
                    ok = False
                    break
                r = value // b
            steps.append((value, op, b, r))
            value = r
        if ok and abs(value) <= 10_000:
            return steps, value


def generate_chained_corpus(n_examples: int, seed: int = 0) -> list:
    rng = random.Random(seed)
    examples = []
    for _ in range(n_examples):
        n_steps = rng.randint(2, 4)
        steps, final = make_chained_expression(rng, n_steps)
        examples.append({"text": render_trace(steps, final), "final": final})
    return examples


def generate_corpus(n_examples: int, seed: int = 0) -> list:
    rng = random.Random(seed)
    examples = []
    for _ in range(n_examples):
        n_operands = rng.randint(2, 5)
        _, _, steps, final = make_expression(rng, n_operands)
        examples.append({"text": render_trace(steps, final),
                          "op_sequence": op_sequence(steps), "final": final})
    return examples


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    examples = generate_corpus(n_examples=4000, seed=0)
    split = int(len(examples) * 0.9)
    train, val = examples[:split], examples[split:]

    (OUT_DIR / "train.txt").write_text("".join(e["text"] for e in train))
    (OUT_DIR / "val.txt").write_text("".join(e["text"] for e in val))
    (OUT_DIR / "op_sequences.json").write_text(
        json.dumps([{"op_sequence": e["op_sequence"], "final": e["final"]} for e in examples], indent=2)
    )
    print(f"generated {len(examples)} examples ({len(train)} train / {len(val)} val)")
    print(f"train.txt: {(OUT_DIR / 'train.txt').stat().st_size} bytes, "
          f"val.txt: {(OUT_DIR / 'val.txt').stat().st_size} bytes")
    print("sample:\n" + examples[0]["text"])

    chained = generate_chained_corpus(n_examples=4000, seed=1)  # distinct seed -- independent corpus, not overlapping
    c_split = int(len(chained) * 0.9)
    c_train, c_val = chained[:c_split], chained[c_split:]
    (OUT_DIR / "chained_train.txt").write_text("".join(e["text"] for e in c_train))
    (OUT_DIR / "chained_val.txt").write_text("".join(e["text"] for e in c_val))
    print(f"\ngenerated {len(chained)} chained examples ({len(c_train)} train / {len(c_val)} val)")
    print(f"chained_train.txt: {(OUT_DIR / 'chained_train.txt').stat().st_size} bytes, "
          f"chained_val.txt: {(OUT_DIR / 'chained_val.txt').stat().st_size} bytes")
    print("chained sample:\n" + chained[0]["text"])


if __name__ == "__main__":
    main()
