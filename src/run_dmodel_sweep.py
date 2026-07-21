"""Does raw width alone fix a fixed small corpus, without more data?
Direct test of the double-descent/grokking hypothesis (Power et al. 2022,
arXiv:2201.02177; Nakkiran et al. 2019, arXiv:1912.02292) against this
project's own Chinchilla-ratio-matching precedent (Phase J) -- isolated
to ONE variable: d_model, on rj (~150KB), holding n_layer=4/n_head=4/
attention_layers=(3,) fixed at rj_base_m's own established depth, so
d_model=128 is directly comparable to that already-recorded checkpoint,
not a fresh baseline.

Reuses train.run(args) directly (the real, battle-tested training loop --
patience, periodic sample generation, checkpoint saving) via temporary
in-memory SIZES entries, rather than reimplementing training. No changes
to train.py itself.

Adaptive stopping, per the standing no_scaleup_without_proof rule:
d_model doubling can cost meaningfully more on CPU (compute scales
roughly d_model^2 per layer). Runs sizes in increasing order and stops
BEFORE starting any run whose cost, extrapolated from the measured
wall-clock trend so far, would exceed ~15-20 minutes -- reports however
far the sweep actually gets, honestly.

Usage: python3 src/run_dmodel_sweep.py
"""
import argparse
import json
import time
from pathlib import Path

import train
from fit_scaling_law import extrapolate, fit_power_law

ROOT = Path(__file__).resolve().parent.parent
D_MODELS = [128, 256, 512, 1024, 2048]
N_LAYER = 4
N_HEAD = 4
ATTENTION_LAYERS = [3]  # last of 4 layers -- matches rj_base_m's own hybrid convention
MAX_NEXT_RUN_SECONDS = 20 * 60  # practical ceiling before starting the NEXT size


def make_args(d_model: int) -> argparse.Namespace:
    size_name = f"wsweep{d_model}"
    train.SIZES[size_name] = dict(d_model=d_model, n_layer=N_LAYER, n_head=N_HEAD)
    return argparse.Namespace(
        dataset="rj", arm="base", size=size_name, block_size=128, batch_size=32,
        steps=1000, patience=5, min_delta=0.0, lr=3e-4, n_future=2, align_weight=0.5,
        log_every=200, sample_every=500, moe_experts=0, moe_top_k=1, bitlinear_experts=False,
        rwkv_hybrid=True, attention_layers=ATTENTION_LAYERS, use_bitlinear=False,
        embedding_rank=0, tie_layers=False, vocab_size=1024, selective_decay=False,
        selective_decay_layers=[], use_width_gating=False, use_halting=False,
        tokenizer_variant="", code_core_weight=0.5, code_synthetic_weight=0.15,
        seed=0, num_threads=8, compile_full_model=False,
    )


def main():
    results = []
    prev_wall_s = None
    for d_model in D_MODELS:
        if prev_wall_s is not None and results:
            prev_d = results[-1]["d_model"]
            ratio = prev_wall_s / results[-2]["measured_wall_s"] if len(results) >= 2 else None
            projected = prev_wall_s * (ratio if ratio else (d_model / prev_d) ** 2)
            if projected > MAX_NEXT_RUN_SECONDS:
                print(f"STOPPING before d_model={d_model}: projected ~{projected:.0f}s "
                      f"exceeds the {MAX_NEXT_RUN_SECONDS}s practical ceiling "
                      f"(based on the measured trend so far).")
                break

        args = make_args(d_model)
        print(f"\n=== d_model={d_model} ===")
        t0 = time.time()
        result = train.run(args)
        # Measured independently, not read from result["wall_s"] -- that key
        # gets overwritten by best_entry's per-checkpoint wall-clock (the
        # time AT the best step, not the total run including any patience
        # overrun), which would make the cost-projection below unreliable.
        measured_wall_s = round(time.time() - t0, 2)
        result["measured_wall_s"] = measured_wall_s
        results.append({"d_model": d_model, **result})
        prev_wall_s = measured_wall_s

    (ROOT / "runs" / "dmodel_sweep_results.json").write_text(json.dumps(results, indent=2))
    print(f"\nsaved -> {ROOT / 'runs' / 'dmodel_sweep_results.json'}")

    if len(results) >= 2:
        points = [(r["n_params"], r["val_loss"]) for r in results]
        fit = fit_power_law(points)
        print(f"\nfit_scaling_law (assumes monotonic decrease -- see caveat if the "
              f"table above isn't monotonic): a={fit['a']:.4f} alpha={fit['alpha']:.4f} "
              f"n_points={fit['n_points']}")

    print("\n=== summary table ===")
    for r in results:
        print(f"d_model={r['d_model']:>5}  n_params={r['n_params']:>10,}  "
              f"best_val={r['val_loss']:.4f}  wall_s={r['measured_wall_s']:.1f}")
    return results


if __name__ == "__main__":
    import resource
    main()
    print(f"peak_rss_mb: {resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024:.1f}")
