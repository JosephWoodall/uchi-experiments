"""Outer-loop hyperparameter search over the training recipe itself
(lr / beta2 / weight_decay / plateau schedule knobs), not architecture --
the natural next step after --nanogpt-recipe (train.py) showed those values
being hand-copied from nanoGPT/uchi rather than tuned for Ducky's own toy
corpora. Small-scale-first, per tasks/core_principle.md: cheap toy trials
(rj/m, a few minutes each) here, not a search at real (chinchilla_min/xl)
scale, matching every other sweep this project has run.

Random search, not grid or Bayesian: Bergstra & Bengio 2012 ("Random
Search for Hyper-Parameter Optimization," JMLR 13) show random sampling
dominates grid search when only a few dimensions actually matter (true
here -- lr and weight_decay are known to matter far more than the
plateau-cooldown knob), and it needs no new dependency (this project's
own established preference -- see graph.py's plain-dict graph instead of
Neo4j/FAISS) over something like Optuna's TPE sampler, which isn't
installed and isn't obviously worth the added surface for a 20-trial toy
sweep.

Uses --lr-schedule plateau (not cosine) deliberately: cosine needs a
max_steps horizon guessed in advance, and guessing wrong produced two
false negatives already this session (rj @ 700 steps, code @ 1200 steps
-- see tasks/todo.md Phase X). Plateau reacts to the model's own live
val-loss trend, so a short toy trial and a long real run don't need
different schedule tuning.

Usage:
  python3 src/run_hpo_sweep.py
  python3 src/run_hpo_sweep.py --trials 30 --steps 1000
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from train import RUNS_DIR, build_parser, run

SEARCH_SPACE = {
    # (low, high, "log" | "linear") for continuous; explicit list for discrete
    "lr": (1e-4, 1e-3, "log"),
    "beta2": (0.90, 0.999, "linear"),
    "weight_decay": (0.01, 0.3, "log"),
    "plateau_patience": [2, 3, 4, 5, 6],
    "plateau_factor": (0.3, 0.7, "linear"),
}

# The exact hand-copied nanoGPT/uchi values --nanogpt-recipe defaults to --
# included as trial 0 so the sweep's own "best found" has a fixed reference
# point to beat, not just a ranking among itself.
DEFAULT_RECIPE = {"lr": 3e-4, "beta2": 0.95, "weight_decay": 0.1,
                   "plateau_patience": 5, "plateau_factor": 0.5}


def sample_config(rng: np.random.Generator) -> dict:
    cfg = {}
    for name, spec in SEARCH_SPACE.items():
        if isinstance(spec, list):
            cfg[name] = spec[rng.integers(0, len(spec))]
        else:
            low, high, scale = spec
            if scale == "log":
                cfg[name] = float(np.exp(rng.uniform(np.log(low), np.log(high))))
            else:
                cfg[name] = float(rng.uniform(low, high))
    return cfg


def run_trial(cfg: dict, args_ns) -> dict:
    # Every trial shares one run_dir (hyperparameter values aren't part of
    # train.py's run-name suffix scheme) -- each trial's model_best.pt/
    # metrics.json overwrites the last one's. Deliberate here, not the
    # overwrite-collision bug class this project has hit and fixed before
    # (MoE/rwkv/seed suffixes): this sweep's record of truth is results.json
    # (val_loss/best_step per trial, captured inline below before the next
    # trial overwrites the checkpoint), not N throwaway toy checkpoints.
    p = build_parser()
    argv = [
        "--dataset", args_ns.dataset, "--arm", "base", "--size", args_ns.size,
        "--steps", str(args_ns.steps), "--seed", str(args_ns.seed),
        "--vocab-size", str(args_ns.vocab_size),
        "--log-every", str(args_ns.log_every), "--sample-every", str(args_ns.steps + 1),
        "--nanogpt-recipe", "--lr-schedule", "plateau",
        "--lr", str(cfg["lr"]), "--beta2", str(cfg["beta2"]),
        "--weight-decay", str(cfg["weight_decay"]),
        "--plateau-patience", str(cfg["plateau_patience"]),
        "--plateau-factor", str(cfg["plateau_factor"]),
    ]
    args = p.parse_args(argv)
    torch.manual_seed(args.seed)
    result = run(args)
    return {"val_loss": result["val_loss"], "best_step": result["best_step"], "n_params": result["n_params"]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="rj")
    ap.add_argument("--size", default="m")
    ap.add_argument("--steps", type=int, default=800)
    ap.add_argument("--vocab-size", type=int, default=1024)
    ap.add_argument("--log-every", type=int, default=100)
    ap.add_argument("--seed", type=int, default=57, help="fixed training seed shared by every "
                     "trial -- only the hyperparameters vary, matching this project's own "
                     "seed-controlled A/B convention")
    ap.add_argument("--trials", type=int, default=20)
    ap.add_argument("--sample-seed", type=int, default=42, help="RNG seed for the search itself, "
                     "so the sweep's own trial sequence is reproducible")
    args_ns = ap.parse_args()

    rng = np.random.default_rng(args_ns.sample_seed)
    trials = [dict(DEFAULT_RECIPE)] + [sample_config(rng) for _ in range(args_ns.trials)]

    results = []
    t0 = time.time()
    for i, cfg in enumerate(trials):
        label = "default_recipe" if i == 0 else f"trial_{i}"
        print(f"[{i + 1}/{len(trials)}] {label}: {cfg}")
        try:
            metrics = run_trial(cfg, args_ns)
        except Exception as e:
            print(f"  FAILED: {e}")
            continue
        entry = {"label": label, **cfg, **metrics}
        results.append(entry)
        print(f"  -> val_loss={metrics['val_loss']:.4f} @ step {metrics['best_step']}")

    results.sort(key=lambda r: r["val_loss"])
    out_dir = RUNS_DIR / f"hpo_sweep_{args_ns.dataset}_{args_ns.size}"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "results.json").write_text(json.dumps(results, indent=2))

    default_entry = next((r for r in results if r["label"] == "default_recipe"), None)
    print(f"\ndone in {time.time() - t0:.1f}s, {len(results)}/{len(trials)} trials completed")
    print(f"results -> {out_dir / 'results.json'}")
    print("\ntop 5:")
    for r in results[:5]:
        print(f"  {r['label']}: val_loss={r['val_loss']:.4f} lr={r['lr']:.2e} beta2={r['beta2']:.3f} "
              f"wd={r['weight_decay']:.3f} patience={r['plateau_patience']} factor={r['plateau_factor']:.2f}")
    if default_entry is not None and results:
        best = results[0]
        gap = default_entry["val_loss"] - best["val_loss"]
        print(f"\ndefault_recipe (hand-copied nanoGPT/uchi values): val_loss={default_entry['val_loss']:.4f}")
        print(f"best found by search: val_loss={best['val_loss']:.4f} ({'-' if gap >= 0 else '+'}{abs(gap):.4f} "
              f"vs default_recipe)")


if __name__ == "__main__":
    main()
