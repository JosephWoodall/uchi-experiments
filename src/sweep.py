"""Runs the full ablation matrix (sizes x arms x datasets, respecting the
jepa-aux code-only constraint) and fits a power-law scaling curve per
(dataset, arm) so arms can be compared at matched params, not by vibes.
"""
import argparse
import csv
import time
from pathlib import Path

import numpy as np

import train

ROOT = Path(__file__).resolve().parent.parent
RESULTS_CSV = ROOT / "runs" / "results.csv"

MATRIX = {
    "rj": ["base", "mtp"],
    "code": ["base", "mtp", "jepa-aux"],
}


def default_args(dataset, arm, size, steps, moe_experts=0, moe_top_k=1, bitlinear_experts=False,
                  rwkv_hybrid=False, attention_layers=None, use_bitlinear=False, embedding_rank=0,
                  seed=0, num_threads=8, compile_full_model=False, patience=0, min_delta=0.0):
    checkpoint_every = max(steps // 5, 1)
    return argparse.Namespace(
        dataset=dataset,
        arm=arm,
        size=size,
        block_size=128,
        batch_size=32 if arm != "jepa-aux" else 16,
        steps=steps,
        patience=patience,
        min_delta=min_delta,
        lr=3e-4,
        n_future=2,
        align_weight=0.5,
        log_every=checkpoint_every,
        sample_every=checkpoint_every,
        moe_experts=moe_experts,
        moe_top_k=moe_top_k,
        bitlinear_experts=bitlinear_experts,
        rwkv_hybrid=rwkv_hybrid,
        attention_layers=attention_layers or [],
        use_bitlinear=use_bitlinear,
        embedding_rank=embedding_rank,
        seed=seed,
        num_threads=num_threads,
        compile_full_model=compile_full_model,
    )


def loss_key(row):
    """Different arms log different metric names; normalize to one number
    for the scaling-law fit. Always prefer held-out val loss over train
    loss -- jepa-aux's train loss is measured on ~90 examples revisited
    dozens of times per run and is not comparable across arms."""
    return row.get("val_loss", row.get("val_code_lm_loss"))


def fit_power_law(sizes_params, losses):
    """log(L) = log(a) - b*log(N); returns (a, b)."""
    x = np.log(sizes_params)
    y = np.log(losses)
    A = np.vstack([-x, np.ones_like(x)]).T
    coeffs, *_ = np.linalg.lstsq(A, y, rcond=None)
    b, log_a = coeffs
    return float(np.exp(log_a)), float(b)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--steps", type=int, default=150)
    p.add_argument("--sizes", nargs="+", default=["xs", "s", "m"])  # 'l' (~2.5min/run) is opt-in, not default
    args = p.parse_args()

    results = []
    t0 = time.time()
    for dataset, arms in MATRIX.items():
        for arm in arms:
            for size in args.sizes:
                run_args = default_args(dataset, arm, size, args.steps)
                print(f"\n=== {dataset}/{arm}/{size} ===")
                summary = train.run(run_args)
                summary["final_loss"] = loss_key(summary)
                results.append(summary)

    RESULTS_CSV.parent.mkdir(parents=True, exist_ok=True)
    with RESULTS_CSV.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=sorted({k for r in results for k in r}))
        writer.writeheader()
        writer.writerows(results)
    print(f"\nsweep done in {time.time() - t0:.1f}s -> {RESULTS_CSV}")

    print("\n=== scaling law fits: L(N) = a * N^-b ===")
    for dataset, arms in MATRIX.items():
        for arm in arms:
            rows = [r for r in results if r["dataset"] == dataset and r["arm"] == arm]
            rows.sort(key=lambda r: r["n_params"])
            if len(rows) < 2:
                continue
            n = np.array([r["n_params"] for r in rows], dtype=float)
            l = np.array([r["final_loss"] for r in rows], dtype=float)
            a, b = fit_power_law(n, l)
            points = list(zip(n.astype(int).tolist(), l.round(3).tolist()))
            print(f"{dataset}/{arm}: a={a:.3f} b={b:.4f}  points={points}")


if __name__ == "__main__":
    main()
