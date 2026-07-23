"""Re-run just the m/l points of run_scaling_sweep.py's 4x3x2 sweep, under
the fixed stopping rule (PATIENCE=6, VAL_BATCHES=10 -- see run_scaling_sweep.py's
comment on why the original PATIENCE=2/5-batch settings produced a false-plateau
artifact at m/l, confirmed on all 3 domains). xs/s were not showing the
stuck-early pattern and are reused as-is from their already-completed
runs/sweep_{domain}_{size}_{arch}/config.json files -- no need to re-train
what wasn't contaminated.

Overwrites the old (contaminated) sweep_{domain}_{m,l}_{arch} run dirs, since
those numbers are superseded, not a second reference point to keep.
"""
import json

from run_scaling_sweep import (
    ARCHITECTURES,
    DOMAINS,
    RUNS_DIR,
    load_domain,
    train_one,
)
from fit_scaling_law import compare_architectures
from tokenizer import Tokenizer

RESIZED_SIZES = ["m", "l"]
REUSED_SIZES = ["xs", "s"]


def load_reused(domain: str, size: str, arch: str) -> dict:
    cfg_path = RUNS_DIR / f"sweep_{domain}_{size}_{arch}" / "config.json"
    return json.loads(cfg_path.read_text())


def main():
    tok = Tokenizer(vocab_size=32768, variant="balanced")
    results = []

    for size in REUSED_SIZES:
        for domain in DOMAINS:
            for arch in ARCHITECTURES:
                results.append(load_reused(domain, size, arch))

    for domain in DOMAINS:
        train_ids, val_ids = load_domain(domain, tok)
        for size in RESIZED_SIZES:
            for arch in ARCHITECTURES:
                label = f"{domain}/{size}/{arch}"
                print(f"=== {label} (patience=6, val_batches=10) ===", flush=True)
                r = train_one(domain, size, arch, tok, train_ids, val_ids)
                results.append(r)
                print(f"RESULT {label}:", r, flush=True)

    print("\n=== ALL RESULTS (xs/s reused, m/l re-trained) ===")
    for r in results:
        print(r)

    print("\n=== PER-DOMAIN DECISION (block_stack_params as N) ===")
    verdicts = {}
    for domain in DOMAINS:
        points_by_arch = {
            arch: [
                (r["block_stack_params"], r["best_val"])
                for r in results if r["domain"] == domain and r["arch"] == arch
            ]
            for arch in ARCHITECTURES
        }
        target_n = max(p[0] for pts in points_by_arch.values() for p in pts) * 10
        comparison = compare_architectures(points_by_arch, target_n)
        verdicts[domain] = comparison
        print(f"{domain}: {json.dumps(comparison, indent=2)}")

    all_favor_selective = all(v["winner"] == "selective" for v in verdicts.values())
    print(f"\n=== FINAL VERDICT: promote selective decay = {all_favor_selective} ===")

    (RUNS_DIR / "scaling_sweep_v2_results.json").write_text(json.dumps({
        "results": results, "verdicts": verdicts,
    }, indent=2))

    return results, verdicts


if __name__ == "__main__":
    import resource
    results, verdicts = main()
    print(f"\npeak_rss_mb: {resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024:.1f}")
