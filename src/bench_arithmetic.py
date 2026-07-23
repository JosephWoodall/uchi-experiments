"""Real, executable-ground-truth benchmark for calculator-grounded
generation -- same discipline as bench_ducky.py: real known answers,
not vibes. Scope matches this round's confirmed decision (pure numeric
expressions, not word problems): each task is a single prompt ending
right at "=" with a real, pre-computed correct result, so grading is a
direct read of the first stated numeral against ground truth -- no
ambiguity about which sub-expression the model "should" have produced,
unlike a multi-step chain where later operands are independent random
draws the model has no way to predict.

Runs both generate_with_grounding (plain -- the "mimicry only" baseline)
and generate_with_calculator (the fix) from the identical prompt/model/
thresholds, reporting final-answer accuracy side by side.

Usage: python3 src/bench_arithmetic.py
"""
import json
import random
import re
from pathlib import Path

import torch

from grounding import evaluate_arithmetic
from inference import generate_with_calculator, generate_with_grounding
from model import GPTConfig, TinyGPT
from tokenizer import Tokenizer
from train import SIZES

ROOT = Path(__file__).resolve().parent.parent
RUN_DIR = ROOT / "runs" / "arithmetic_base_s_rwkv"
CHAINED_RUN_DIR = ROOT / "runs" / "arithmetic_chained_base_s_rwkv"
N_TASKS = 20
BENCH_SEED = 12345  # distinct from generate_arithmetic_corpus.py's training seed (0) -- genuinely held out
CHAIN_SEED = 54321
_LEADING_NUMBER = re.compile(r"\s*(-?\d+(?:\.\d+)?)")
_STEP_RE = re.compile(r"(-?\d+(?:\.\d+)?)\s*([+\-*/])\s*(-?\d+(?:\.\d+)?)\s*=\s*(-?\d+(?:\.\d+)?)")
_ANSWER_RE = re.compile(r"Answer:\s*(-?\d+(?:\.\d+)?)")


def make_tasks(n: int, seed: int) -> list:
    """Single binary-op expressions (a op b), matching the training
    corpus's own "Step 1: a op b = r" line shape exactly -- multi-operand
    chains aren't a fair target here, since later operands in the training
    format are independent random draws no model could predict from
    context; a single expression's result is the one well-posed,
    gradeable claim in scope this round.
    """
    rng = random.Random(seed)
    ops = ["+", "-", "*", "/"]
    tasks = []
    while len(tasks) < n:
        a, b = rng.randint(1, 20), rng.randint(1, 20)
        op = rng.choice(ops)
        expr = f"{a} {op} {b}"
        real = evaluate_arithmetic(expr)
        if op == "/" and (b == 0 or a % b != 0):
            continue  # keep results clean integers, matching training corpus convention
        tasks.append({"expr": expr, "expected": real})
    return tasks


def grade(generated_text: str, expected) -> bool:
    m = _LEADING_NUMBER.match(generated_text)
    if not m:
        return False
    try:
        return abs(float(m.group(1)) - float(expected)) < 1e-6
    except ValueError:
        return False


def load_checkpoint(run_dir: Path = RUN_DIR):
    cfg_dict = json.loads((run_dir / "config.json").read_text())
    tok = Tokenizer(vocab_size=cfg_dict["vocab_size"])
    size_cfg = SIZES[cfg_dict["size"]]
    cfg = GPTConfig(vocab_size=tok.vocab_size, block_size=cfg_dict["block_size"],
                     use_rwkv_hybrid=True, attention_layers=tuple(cfg_dict["attention_layers"]), **size_cfg)
    model = TinyGPT(cfg)
    model.load_state_dict(torch.load(run_dir / "model_best.pt", map_location="cpu"))
    model.eval()
    return model, tok


def analyze_chain(text: str):
    """Extracts every "a op b = c" claim and the final "Answer: N" from
    freely-generated multi-step text. Used only against
    generate_with_calculator's output, where every extracted (a,op,b,c)
    is guaranteed arithmetically correct by construction (splice) -- so
    what's actually being tested is CONTINUITY (does the model's own next
    step start from the real prior result) and SELF-CONSISTENCY (does its
    stated final answer match its own last real computation), not
    arithmetic correctness itself (already solved by Part A).

    Real gap found empirically while building this: there's no stopping
    criterion for this domain (unlike code's is_complete_statement), so
    generation regularly runs past "Answer: N" into an unrelated new
    trace within the same token budget. Steps are extracted only from the
    text UP TO AND INCLUDING the first "Answer:" -- otherwise a spurious
    later trace's steps get compared against this trace's stated answer,
    which isn't a real self-consistency failure, just leftover text.
    """
    m = _ANSWER_RE.search(text)
    scoped_text = text[: m.end()] if m else text
    steps = [(float(a), op, float(b), float(c)) for a, op, b, c in _STEP_RE.findall(scoped_text)]
    answer = float(m.group(1)) if m else None
    return steps, answer


def make_chain_tasks(n: int, seed: int) -> list:
    rng = random.Random(seed)
    tasks = []
    while len(tasks) < n:
        a, b = rng.randint(1, 20), rng.randint(1, 20)
        op = rng.choice(["+", "-", "*", "/"])
        if op == "/" and (b == 0 or a % b != 0):
            continue
        tasks.append(f"Step 1: {a} {op} {b} =")
    return tasks


def bench_chain_coherence(model, tok, n_tasks: int = N_TASKS, max_new_tokens: int = 60) -> dict:
    """Does calculator-grounded free-running generation carry a real prior
    result forward as the NEXT step's own first operand, and does its
    stated final answer match its own last real computation? Both
    well-posed regardless of which operations the model freely chooses
    (unlike trying to match one predetermined multi-step chain, which
    isn't well-posed -- later operands are inherently unpredictable from
    context in this domain).
    """
    prompts = make_chain_tasks(n_tasks, CHAIN_SEED)
    n_continuity_checked, n_continuity_ok = 0, 0
    n_final_checked, n_final_ok = 0, 0
    rows = []
    for prompt in prompts:
        r = generate_with_calculator(model, tok, prompt, max_new_tokens)
        full_text = prompt + " " + r["generated_text"]
        steps, answer = analyze_chain(full_text)
        for i in range(1, len(steps)):
            n_continuity_checked += 1
            n_continuity_ok += int(steps[i][0] == steps[i - 1][3])
        final_ok = None
        if steps and answer is not None:
            n_final_checked += 1
            final_ok = abs(answer - steps[-1][3]) < 1e-6
            n_final_ok += int(final_ok)
        rows.append({"prompt": prompt, "generated": r["generated_text"],
                      "n_steps": len(steps), "final_answer_self_consistent": final_ok})
    return {
        "n_tasks": n_tasks,
        "continuity_rate": n_continuity_ok / n_continuity_checked if n_continuity_checked else None,
        "continuity_checked": n_continuity_checked,
        "final_answer_self_consistent_rate": n_final_ok / n_final_checked if n_final_checked else None,
        "final_answer_checked": n_final_checked,
        "rows": rows,
    }


def main():
    model, tok = load_checkpoint()
    tasks = make_tasks(N_TASKS, BENCH_SEED)

    plain_correct = 0
    calc_correct = 0
    rows = []
    for t in tasks:
        prompt = f"Step 1: {t['expr']} ="
        r_plain = generate_with_grounding(model, tok, prompt, 12, "arithmetic")
        r_calc = generate_with_calculator(model, tok, prompt, 12)
        plain_ok = grade(r_plain["generated_text"], t["expected"])
        calc_ok = grade(r_calc["generated_text"], t["expected"])
        plain_correct += plain_ok
        calc_correct += calc_ok
        rows.append({"expr": t["expr"], "expected": t["expected"],
                      "plain_generated": r_plain["generated_text"], "plain_correct": plain_ok,
                      "calc_generated": r_calc["generated_text"], "calc_correct": calc_ok,
                      "n_splices": len(r_calc["splices"])})

    result = {
        "n_tasks": N_TASKS,
        "plain_accuracy": plain_correct / N_TASKS,
        "calculator_accuracy": calc_correct / N_TASKS,
        "rows": rows,
    }
    print("=== single-expression benchmark (original checkpoint) ===")
    print(json.dumps({k: v for k, v in result.items() if k != "rows"}, indent=2))

    print("\n=== chain coherence, original (independent-operand) checkpoint ===")
    orig_chain = bench_chain_coherence(model, tok)
    print(json.dumps({k: v for k, v in orig_chain.items() if k != "rows"}, indent=2))

    chained_model, chained_tok = load_checkpoint(CHAINED_RUN_DIR)
    print("\n=== chain coherence, chained-trained checkpoint ===")
    chained_chain = bench_chain_coherence(chained_model, chained_tok)
    print(json.dumps({k: v for k, v in chained_chain.items() if k != "rows"}, indent=2))

    return {"single_expression": result, "chain_original": orig_chain, "chain_chained_trained": chained_chain}


if __name__ == "__main__":
    main()
