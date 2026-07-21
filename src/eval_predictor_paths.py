"""Tests whether UniversalPredictor (sequence_predictor.py) can predict
Ducky's own discrete DECISION streams above a majority-class baseline --
not tokens (that's TinyGPT's job), the sequential control-flow symbols
Ducky already emits as a byproduct of running. Three paths, per the
user's explicit "test them all" rather than just the recommended one.
Mirrors eval_step_sequencer.py's exact evaluation methodology (majority-
class baseline, UniversalPredictor.train()/predict()/feedback() cycle,
held-out split) -- same harness, applied to real Ducky decision streams
instead of a synthetic arithmetic corpus.

Path 1 (Confidence Watchdog): fast/slow/abstain per generated token,
walking real corpus text -- the busiest, most naturally-ordered stream
Ducky has (new data every token).
Path 2 (Repair Strategy Advisor): per-attempt outcome (empty/fail/ok)
across generate_with_repair's retry sequences -- much sparser, <=4
symbols per call, only ~10 calls available (bench_ducky's task set).
Path 3 (Session Pattern Watcher): per-turn answered/abstained outcome
across simulated multi-turn sessions -- sparsest of the three.

Usage: python3 src/eval_predictor_paths.py
"""
import json
from collections import Counter

import torch

from bench_ducky import TASKS
from data import load_lm_corpus
from ducky import Ducky
from inference import ABSTAIN, predict_next
from repair_loop import generate_with_repair
from sequence_predictor import UniversalPredictor


def majority_baseline(train_seqs: list, val_seqs: list) -> dict:
    counts = Counter()
    for seq in train_seqs:
        for s in seq[1:]:  # position 0 has no prior context, excluded like the real eval loops
            counts[s] += 1
    if not counts:
        return {"accuracy": 0.0, "n_total": 0}
    majority = counts.most_common(1)[0][0]
    n_correct, n_total = 0, 0
    for seq in val_seqs:
        for i in range(1, len(seq)):
            n_total += 1
            n_correct += int(seq[i] == majority)
    return {"majority": majority, "accuracy": n_correct / n_total if n_total else 0.0, "n_total": n_total}


def eval_universal_predictor(train_seqs: list, val_seqs: list, context_length: int = 8) -> dict:
    predictor = UniversalPredictor(context_length=context_length)
    for seq in train_seqs:
        predictor.history = []  # fresh context per sequence, trie/credibilities persist across sequences
        predictor.train(seq)

    n_correct, n_total = 0, 0
    for seq in val_seqs:
        predictor.history = []
        for i, sym in enumerate(seq):
            if i > 0:
                pred, _ = predictor.predict()
                n_total += 1
                n_correct += int(pred == sym)
            predictor.observe(sym)
            predictor.feedback(sym)
    return {"accuracy": n_correct / n_total if n_total else 0.0, "n_total": n_total}


def path1_confidence_watchdog(d: Ducky, n_steps: int = 600) -> tuple:
    """One long fast/slow/abstain sequence from walking real corpus text
    token-by-token through predict_next -- the densest stream available.
    """
    train_ids, val_ids = load_lm_corpus(d.domain, d.tok)
    ids = torch.cat([train_ids, val_ids])
    block = d.model.cfg.block_size
    n_steps = min(n_steps, len(ids) - block - 1)
    symbols = []
    for start in range(n_steps):
        ctx = ids[start:start + block].unsqueeze(0)
        next_id, info = predict_next(d.model, d.graph, ctx, d.fast_t, d.abstain_t, d.slow_abstain_t,
                                      ngram_index=d.ngram_index)
        symbols.append("abstain" if next_id == ABSTAIN else info["path"])
    split = int(len(symbols) * 0.8)
    return [symbols[:split]], [symbols[split:]]


def _repair_attempt_symbol(attempt: dict, domain: str) -> str:
    if attempt["n_tokens_generated"] == 0:
        return "empty"
    ok = (domain != "code" or attempt.get("syntax_valid", True)) and \
         (attempt.get("self_critique_score") or 0.0) >= 0.0
    return "ok" if ok else "fail"


def path2_repair_advisor(d: Ducky) -> tuple:
    """Per-attempt outcome sequence across generate_with_repair calls on
    bench_ducky's 10 real tasks -- one sequence per call (<=4 symbols),
    train/val split across calls (not within one, unlike Path 1/3).
    """
    sequences = []
    for task in TASKS:
        result = generate_with_repair(
            d.model, d.tok, d.graph, task["prompt"], d.max_new_tokens, domain=d.domain,
            fast_threshold=d.fast_t, abstain_threshold=d.abstain_t, slow_abstain_threshold=d.slow_abstain_t,
            symbol_table=d.symbol_table, ngram_index=d.ngram_index,
        )
        seq = [_repair_attempt_symbol(a, d.domain) for a in result["attempts_log"]]
        if len(seq) >= 2:  # need at least 2 symbols for a next-symbol prediction to exist
            sequences.append(seq)
    split = max(1, int(len(sequences) * 0.8))
    return sequences[:split], sequences[split:]


def path3_session_watcher(n_sessions: int = 12, turns_per_session: int = 5) -> tuple:
    """Per-turn answered/abstained outcome across simulated multi-turn
    sessions -- deliberately mixes clearly in-domain (rj corpus openings)
    and out-of-domain (single common words) prompts so the stream isn't
    trivially all-one-class, which would make the majority baseline win
    by construction rather than by real difficulty.
    """
    in_domain = ["ROMEO:", "JULIET:", "NURSE.", "BENVOLIO.", "MERCUTIO."]
    out_domain = ["The weather", "banana", "xyzzy plonk", "1234567", "quantum"]
    prompts = (in_domain + out_domain) * 2  # 10 prompts, cycle to fill turns_per_session

    sequences = []
    for s in range(n_sessions):
        d = Ducky(domain="rj", backbone="hybrid", track_history=True, use_cache=True)
        seq = []
        for t in range(turns_per_session):
            prompt = prompts[(s * turns_per_session + t) % len(prompts)]
            response = d.ask(prompt)
            seq.append("answered" if response else "abstained")
        sequences.append(seq)
    split = int(n_sessions * 0.8)
    return sequences[:split], sequences[split:]


def run_path(name: str, train_seqs: list, val_seqs: list) -> dict:
    n_train_syms = sum(len(s) for s in train_seqs)
    n_val_syms = sum(len(s) for s in val_seqs)
    baseline = majority_baseline(train_seqs, val_seqs)
    up = eval_universal_predictor(train_seqs, val_seqs)
    result = {
        "path": name, "n_train_sequences": len(train_seqs), "n_val_sequences": len(val_seqs),
        "n_train_symbols": n_train_syms, "n_val_symbols": n_val_syms,
        "majority_baseline": baseline, "universal_predictor": up,
        "delta": round(up["accuracy"] - baseline["accuracy"], 4),
    }
    print(json.dumps(result, indent=2))
    return result


def main():
    print("=== Path 1: Confidence Watchdog (rj, fast/slow/abstain per token) ===")
    d_rj = Ducky(domain="rj", backbone="hybrid", use_cache=True)
    train1, val1 = path1_confidence_watchdog(d_rj)
    r1 = run_path("confidence_watchdog", train1, val1)

    print("\n=== Path 2: Repair Strategy Advisor (code, per-attempt outcome) ===")
    d_code = Ducky(domain="code", backbone="hybrid", use_cache=True)
    train2, val2 = path2_repair_advisor(d_code)
    r2 = run_path("repair_advisor", train2, val2)

    print("\n=== Path 3: Session Pattern Watcher (rj, per-turn answered/abstained) ===")
    train3, val3 = path3_session_watcher()
    r3 = run_path("session_watcher", train3, val3)

    print("\n=== Summary ===")
    for r in (r1, r2, r3):
        print(f"{r['path']}: baseline={r['majority_baseline']['accuracy']:.3f} "
              f"UP={r['universal_predictor']['accuracy']:.3f} delta={r['delta']:+.4f} "
              f"(n_val_symbols={r['n_val_symbols']})")
    return {"path1": r1, "path2": r2, "path3": r3}


if __name__ == "__main__":
    main()
