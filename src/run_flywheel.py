"""Part B: a grounded self-training "flywheel" -- generate candidates from
the code-domain ensemble (eval_ensemble.py's 3 seed-varied `s`-size
models), verify them against something real (never just "models agree
with each other" -- agreement can mean a shared mistake, not a shared
truth), pool the verified ones into a new training file, retrain, and
re-measure. One round only this pass, per tasks/core_principle.md's
small-scale-first rule -- more rounds are a follow-up gated on this one
showing something real.

Two verification tiers, both real:
  Tier 1: actually passes bench_ducky.py's executable asserts (genuine
    ground truth, strongest possible signal).
  Tier 2 (for cases with no assert to check, or that don't pass): parses
    (grounding.verify_code_syntax) AND is internally arity-consistent
    (grounding.check_call_arity_consistency) -- both already-validated
    signals, not a new heuristic invented for this round.

Deliberately scope-narrowed from the plan's "bench tasks + additional
held-out prompts" to just bench_ducky.py's 10 real tasks: every one of
them has genuine ground truth to check against, avoiding the need to
synthesize new prompts just to have more volume.
"""
import json
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

from bench_ducky import TASKS, run_task
from data import load_lm_corpus
from eval_ensemble import load_model
from grounding import check_call_arity_consistency, has_real_statement, verify_code_syntax
from inference import _block_repeat_ngrams
from tokenizer import Tokenizer

ROOT = Path(__file__).resolve().parent.parent
FLYWHEEL_CORPUS_PATH = ROOT / "data" / "code" / "flywheel_round1.txt"

SEED_RUNS = ["code_base_s_seed57", "code_base_s_seed58", "code_base_s_seed59"]
N_CANDIDATES = 5
MAX_NEW_TOKENS = 48
TEMPERATURE = 0.8


@torch.no_grad()
def ensemble_generate(models: list, tok: Tokenizer, prompt: str, max_new_tokens: int,
                       temperature: float = 0.0, no_repeat_ngram_size: int = 4) -> str:
    """Averages softmax probabilities across all models at every step --
    the most direct realization of "share outputs to build on one
    another": every token decision is already blended across all 3
    members, not decided by one model then handed to the next.
    """
    ids = tok.encode(prompt)
    block_size = models[0].cfg.block_size
    for _ in range(max_new_tokens):
        ctx = torch.tensor([ids[-block_size:]], dtype=torch.long)
        probs_list = []
        for m in models:
            logits, _, _, _ = m(ctx)
            step_logits = logits[0, -1, :]
            if no_repeat_ngram_size > 0:
                step_logits = _block_repeat_ngrams(step_logits, ids, no_repeat_ngram_size)
            probs_list.append(F.softmax(step_logits, dim=-1))
        avg_probs = torch.stack(probs_list).mean(dim=0)
        if temperature <= 0:
            next_id = avg_probs.argmax().item()
        else:
            tempered = torch.softmax(torch.log(avg_probs + 1e-12) / temperature, dim=-1)
            next_id = torch.multinomial(tempered, num_samples=1).item()
        ids.append(next_id)
    generated_ids = ids[len(tok.encode(prompt)):]
    return tok.decode(generated_ids)


def round0_baseline(models: list, tok: Tokenizer) -> dict:
    """Greedy (temperature=0), one shot per task -- the honest baseline,
    matching bench_ducky.py's own established convention.
    """
    def ensemble_ask(prompt):
        return ensemble_generate(models, tok, prompt, MAX_NEW_TOKENS, temperature=0.0)

    ensemble_results = [
        {"name": t["name"], **run_task(t["prompt"] + ensemble_ask(t["prompt"]), t["asserts"])}
        for t in TASKS
    ]
    ensemble_pass = sum(r["passed"] for r in ensemble_results)

    per_model_pass = []
    for m in models:
        def single_ask(prompt, m=m):
            return ensemble_generate([m], tok, prompt, MAX_NEW_TOKENS, temperature=0.0)
        results = [run_task(t["prompt"] + single_ask(t["prompt"]), t["asserts"]) for t in TASKS]
        per_model_pass.append(sum(r["passed"] for r in results))

    return {
        "n_tasks": len(TASKS),
        "ensemble_pass": ensemble_pass,
        "per_model_pass": per_model_pass,
        "best_single_pass": max(per_model_pass),
    }


def generate_and_verify(models: list, tok: Tokenizer) -> dict:
    """For each bench_ducky task, sample N_CANDIDATES from the ensemble at
    temperature>0, keep the first Tier-1 (real assert pass) or Tier-2
    (syntax-valid + arity-consistent) candidate found. Returns the pooled
    verified text plus counts -- the fuel-quantity number is the headline
    result, not just whether the round "succeeded."
    """
    verified_lines = []
    tier1_count, tier2_count = 0, 0

    for task in TASKS:
        found = None
        for _ in range(N_CANDIDATES):
            completion = ensemble_generate(models, tok, task["prompt"], MAX_NEW_TOKENS, temperature=TEMPERATURE)
            full_text = task["prompt"] + completion
            outcome = run_task(full_text, task["asserts"])
            if outcome["passed"]:
                found = ("tier1", full_text)
                break
            # verify_code_syntax alone lets comment-only "bodies" through (a `#` line
            # produces no AST node and isn't a syntax error) -- has_real_statement is
            # the fix, found necessary directly by round 1's output (see grounding.py).
            if verify_code_syntax(full_text) and has_real_statement(full_text):
                arity = check_call_arity_consistency(full_text)
                if arity["consistent"] is not False:  # True or None (couldn't check) both OK, only a real conflict vetoes
                    found = found or ("tier2", full_text)
        if found:
            tier, text = found
            if tier == "tier1":
                tier1_count += 1
            else:
                tier2_count += 1
            verified_lines.append(text)

    return {
        "tier1_count": tier1_count, "tier2_count": tier2_count,
        "total_verified": tier1_count + tier2_count, "n_tasks_attempted": len(TASKS),
        "verified_text": verified_lines,
    }


def main():
    tok = Tokenizer(vocab_size=1024)
    models = [load_model(name) for name in SEED_RUNS]

    print("=== Round 0 baseline (bench_ducky pass rate) ===")
    baseline = round0_baseline(models, tok)
    print(json.dumps(baseline, indent=2))

    print("\n=== Generating + verifying candidates ===")
    fuel = generate_and_verify(models, tok)
    print(json.dumps({k: v for k, v in fuel.items() if k != "verified_text"}, indent=2))

    if fuel["total_verified"] == 0:
        print("\nNo verified fuel produced -- the flywheel has nothing to spin on at this scale. Stopping here.")
        return {"baseline": baseline, "fuel": fuel, "retrained": False}

    FLYWHEEL_CORPUS_PATH.parent.mkdir(parents=True, exist_ok=True)
    FLYWHEEL_CORPUS_PATH.write_text("\n\n".join(fuel["verified_text"]) + "\n")
    print(f"\nWrote {fuel['total_verified']} verified example(s) to {FLYWHEEL_CORPUS_PATH}")
    return {"baseline": baseline, "fuel": fuel, "retrained": False}


if __name__ == "__main__":
    result = main()
    print("\n=== SUMMARY ===")
    print(json.dumps({k: v for k, v in result.items() if k != "fuel"}, indent=2, default=str))
