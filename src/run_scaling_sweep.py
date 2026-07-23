"""24-config scaling sweep to validate selective decay properly across a
real size range (not 2 close-together toy points) before deciding whether
to promote it ahead of the next production retrain -- and the first real
training use of spm_32768_balanced.model (see tasks/ducky.md's
tokenizer-fairness section), incorporating it without a full production
retrain.

4 sizes (xs, s, m, l -- the original xs/s toy pass plus m/l, now that the
GPU that was previously pinned by a live uchi training job is free) x 3
domains (rj, code_core, terminal) x 2 architectures (RWKV-hybrid vs.
selective-hybrid; dense excluded -- already well-established as losing to
hybrid across many prior rounds, not the open question here).
Self-contained training loop (reuses data.get_lm_batch /
train.compute_lm_loss directly) rather than routing through train.py's
--dataset, since "code_core" and "terminal" aren't `load_lm_corpus`'s
domains and this stays a one-off sweep, not a change to train.py's shared
CLI surface.

The original xs/s-only pass's own limitation, now addressed: 2 close
points is a secant slope, not a robust regression. 4 points spanning a
real size range lets fit_scaling_law.py's power-law fit (built generally,
any number of points) actually be trusted as a curve, not just a
direction between two nearby dots.
"""
import json
import time
from pathlib import Path

import torch

from data import ROOT as DATA_ROOT
from data import _tokenize_corpus, get_lm_batch, load_lm_corpus
from fit_scaling_law import compare_architectures
from model import GPTConfig, TinyGPT
from tokenizer import Tokenizer
from train import SIZES, compute_lm_loss

RUNS_DIR = DATA_ROOT / "runs"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
EMBEDDING_RANK = 32  # keeps vocab=32768's embedding table from dwarfing xs/s/m/l's block stack
SEED = 57
STEPS = 2000
# PATIENCE=2 @ 50-step checks (100 steps of grace) + 5-batch val averaging was
# tuned on xs/s only. At m/l it produced a false-plateau artifact: one arch
# would get stuck at step ~150-250 while the other, equally-valid arch kept
# descending to step ~1000-1650 -- a stopping-rule artifact, not a real
# architecture difference (confirmed on rj, code_core, and terminal alike).
# Loosened for m/l reruns: more grace steps before quitting, less noisy each
# check.
PATIENCE = 6
VAL_BATCHES = 10
BLOCK_SIZE = 128
BATCH_SIZE = 32
SIZES_SWEPT = ["xs", "s", "m", "l"]
DOMAINS = ["rj", "code_core", "terminal"]
ARCHITECTURES = ["rwkv", "selective"]


def load_domain(name: str, tok: Tokenizer):
    if name == "rj":
        return load_lm_corpus("rj", tok)
    if name == "code_core":
        text = (DATA_ROOT / "data" / "code" / "corpus_core.txt").read_text()
        ids = _tokenize_corpus(tok, "code_core", text)
    elif name == "terminal":
        text = (DATA_ROOT / "data" / "terminal" / "nl2bash_corpus.txt").read_text()
        ids = _tokenize_corpus(tok, "terminal", text)
    else:
        raise ValueError(name)
    n_val = int(len(ids) * 0.1)
    return ids[:-n_val], ids[-n_val:]


def train_one(domain: str, size: str, arch: str, tok: Tokenizer, train_ids, val_ids) -> dict:
    torch.manual_seed(SEED)
    size_cfg = SIZES[size]
    n_layer = size_cfg["n_layer"]
    attention_layers = (n_layer - 1,)  # last layer attention, rest RWKV/selective -- consistent
    # "mostly recurrent, periodic attention" pattern regardless of size, matching production intent
    cfg = GPTConfig(
        vocab_size=tok.vocab_size,
        block_size=BLOCK_SIZE,
        use_rwkv_hybrid=True,
        attention_layers=attention_layers,
        use_selective_decay=(arch == "selective"),
        embedding_rank=EMBEDDING_RANK,
        **size_cfg,
    )
    model = TinyGPT(cfg).to(DEVICE)
    n_params = model.num_params()
    block_stack_params = sum(p.numel() for p in model.blocks.parameters())
    opt = torch.optim.AdamW(model.parameters(), lr=3e-4)

    best_val, best_step, patience_ctr = float("inf"), 0, 0
    t0 = time.time()
    for step in range(1, STEPS + 1):
        model.train()
        opt.zero_grad()
        x, targets = get_lm_batch(train_ids, BATCH_SIZE, BLOCK_SIZE, 1)
        x, targets = x.to(DEVICE), targets.to(DEVICE)
        loss = compute_lm_loss(model, x, targets, pad_id=0)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()

        if step % 50 == 0 or step == STEPS:
            model.eval()
            with torch.no_grad():
                val_losses = []
                for _ in range(VAL_BATCHES):
                    vx, vt = get_lm_batch(val_ids, BATCH_SIZE, BLOCK_SIZE, 1)
                    vx, vt = vx.to(DEVICE), vt.to(DEVICE)
                    val_losses.append(compute_lm_loss(model, vx, vt, pad_id=0).item())
                losses = val_losses
            val_loss = sum(losses) / len(losses)
            if val_loss < best_val:
                best_val, best_step, patience_ctr = val_loss, step, 0
            else:
                patience_ctr += 1
                if patience_ctr >= PATIENCE:
                    break

    wall_s = round(time.time() - t0, 2)
    run_name = f"sweep_{domain}_{size}_{arch}"
    run_dir = RUNS_DIR / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "config.json").write_text(json.dumps({
        "domain": domain, "size": size, "arch": arch, "seed": SEED,
        "n_params": n_params, "block_stack_params": block_stack_params,
        "best_val": best_val, "best_step": best_step, "wall_s": wall_s,
    }, indent=2))

    return {
        "domain": domain, "size": size, "arch": arch,
        "n_params": n_params, "block_stack_params": block_stack_params,
        "best_val": round(best_val, 4), "best_step": best_step, "wall_s": wall_s,
    }


def main():
    tok = Tokenizer(vocab_size=32768, variant="balanced")
    results = []
    for domain in DOMAINS:
        train_ids, val_ids = load_domain(domain, tok)
        for size in SIZES_SWEPT:
            for arch in ARCHITECTURES:
                label = f"{domain}/{size}/{arch}"
                print(f"=== {label} ===", flush=True)
                r = train_one(domain, size, arch, tok, train_ids, val_ids)
                results.append(r)
                print(f"RESULT {label}:", r, flush=True)

    print("\n=== ALL RESULTS ===")
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
        target_n = max(p[0] for pts in points_by_arch.values() for p in pts) * 10  # extrapolate 10x past 's'
        comparison = compare_architectures(points_by_arch, target_n)
        verdicts[domain] = comparison
        print(f"{domain}: {json.dumps(comparison, indent=2)}")

    all_favor_selective = all(v["winner"] == "selective" for v in verdicts.values())
    print(f"\n=== FINAL VERDICT: promote selective decay = {all_favor_selective} ===")
    return results, verdicts


if __name__ == "__main__":
    import resource
    results, verdicts = main()
    print(f"\npeak_rss_mb: {resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024:.1f}")
