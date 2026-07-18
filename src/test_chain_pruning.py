"""Does pruning generic-bridge/endpoint chains (filter_generic_chains,
synthetic_relations.py) from the synthetic relational training data help
or hurt held-out compositional generalization? Measured redundancy
motivated this: 'open' alone is the bridge in 20.5% of all chains, and
43.8%/46.3% have a builtin bridge/endpoint -- almost half the synthetic
signal is generic I/O/type-checking boilerplate, not code-specific
structure. Compares training on the full (noisy) chain set against the
filtered (denser) set, both evaluated on the SAME held-out chains (drawn
from the filtered/meaningful set, so the test itself measures something
non-trivial), same methodology as the original synthetic-data validation:
teacher-forced log-prob, true completion vs a real wrong distractor.
"""
import argparse
import random

import torch
import torch.nn.functional as F

from model import GPTConfig, TinyGPT
from synthetic_relations import build_call_graph, filter_generic_chains, find_transitive_chains, synthesize_examples
from tokenizer import Tokenizer
from train import SIZES

EVAL_TEMPLATE = "# {a} -> {b} -> "


def held_out_eval(model, tok, held_out_chains, all_fn_names, seed=0):
    rng = random.Random(seed)
    n_correct = 0
    margins = []
    model.eval()
    for a, b, c in held_out_chains:
        distractors = [f for f in all_fn_names if f != c and f != b and f != a]
        if not distractors:
            continue
        wrong = rng.choice(distractors)
        prompt = EVAL_TEMPLATE.format(a=a, b=b)
        prompt_ids = tok.encode(prompt)

        def avg_logprob(continuation: str) -> float:
            cont_ids = tok.encode(continuation)
            if not cont_ids:
                return float("-inf")
            full = torch.tensor([prompt_ids + cont_ids], dtype=torch.long)
            with torch.no_grad():
                logits, _, _, _ = model(full[:, -model.cfg.block_size:])
            n = min(len(cont_ids), logits.size(1) - 1)
            if n <= 0:
                return float("-inf")
            gen_logits = logits[0, -n - 1: -1, :]
            probs = F.softmax(gen_logits, dim=-1)
            ids_t = torch.tensor(cont_ids[-n:])
            return probs[torch.arange(n), ids_t].log().mean().item()

        true_lp = avg_logprob(c)
        wrong_lp = avg_logprob(wrong)
        margins.append(true_lp - wrong_lp)
        if true_lp > wrong_lp:
            n_correct += 1
    n_total = len(margins)
    mean_margin = sum(margins) / n_total if n_total else 0.0
    return n_correct, n_total, mean_margin


def train_model(train_ids, tok, size: str, steps: int, seed: int, block_size: int = 128,
                 batch_size: int = 32, lr: float = 3e-4, log_every: int = 250):
    torch.manual_seed(seed)
    size_cfg = SIZES[size]
    cfg = GPTConfig(vocab_size=tok.vocab_size, block_size=block_size,
                     use_rwkv_hybrid=True, attention_layers=(2,), **size_cfg)
    model = TinyGPT(cfg)
    opt = torch.optim.AdamW(model.parameters(), lr=lr)

    def get_batch():
        idx = torch.randint(0, len(train_ids) - block_size - 1, (batch_size,))
        x = torch.stack([train_ids[i: i + block_size] for i in idx])
        y = torch.stack([train_ids[i + 1: i + block_size + 1] for i in idx])
        return x, y

    for step in range(1, steps + 1):
        model.train()
        x, y = get_batch()
        opt.zero_grad()
        logits, _, _, _ = model(x)
        loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), y.reshape(-1))
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        if step % log_every == 0 or step == steps:
            print(f"  step {step}: loss={loss.item():.3f}")
    return model


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--size", default="m")
    p.add_argument("--steps", type=int, default=2500)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--num-threads", type=int, default=8)
    args = p.parse_args()
    torch.set_num_threads(args.num_threads)

    tok = Tokenizer()
    from pathlib import Path
    ROOT = Path(__file__).resolve().parent.parent
    core_source = (ROOT / "data" / "code" / "corpus_core.txt").read_text()

    call_graph = build_call_graph(core_source)
    all_fn_names = sorted(call_graph.keys())
    full_chains = find_transitive_chains(call_graph)
    filtered_chains = filter_generic_chains(full_chains)
    print(f"full chains: {len(full_chains)}  filtered (non-generic) chains: {len(filtered_chains)} "
          f"({len(filtered_chains)/len(full_chains)*100:.1f}% kept)")

    # Held-out set drawn from the FILTERED (meaningful) chains, shared by
    # both arms -- testing generalization on non-trivial chains specifically,
    # not on "X calls open" trivia neither arm needs help with.
    rng = random.Random(42)
    shuffled_filtered = filtered_chains[:]
    rng.shuffle(shuffled_filtered)
    n_held_out = max(1, len(shuffled_filtered) // 10)
    held_out_chains = shuffled_filtered[:n_held_out]
    held_out_set = set(held_out_chains)

    full_train_chains = [c for c in full_chains if c not in held_out_set]
    filtered_train_chains = [c for c in filtered_chains if c not in held_out_set]
    print(f"held_out={len(held_out_chains)}  full_train={len(full_train_chains)}  "
          f"filtered_train={len(filtered_train_chains)}")

    full_examples = synthesize_examples(full_train_chains, seed=0)
    filtered_examples = synthesize_examples(filtered_train_chains, seed=0)

    # corpus_core.txt only (~11MB stdlib), not the full concatenated
    # corpus.txt (~162MB incl. site-packages) -- this test is about
    # synthetic-chain redundancy, not corpus scale, and loading the full
    # corpus here would force a fresh, memory-heavy full-corpus
    # tokenization pass under vocab=32768 (no cache exists for that
    # combination) that competes hard with the concurrently-running xl
    # retrain for both CPU and RAM (measured: pushed the system to 11GB+
    # swap). core_source is already read above to build the call graph.
    code_train_ids = torch.tensor(tok.encode(core_source), dtype=torch.long)
    full_synthetic_ids = torch.tensor(tok.encode("\n".join(full_examples)), dtype=torch.long)
    filtered_synthetic_ids = torch.tensor(tok.encode("\n".join(filtered_examples)), dtype=torch.long)

    full_ids = torch.cat([code_train_ids, full_synthetic_ids])
    filtered_ids = torch.cat([code_train_ids, filtered_synthetic_ids])
    print(f"full arm training tokens={len(full_ids):,}  filtered arm training tokens={len(filtered_ids):,}")

    print("\n=== training arm A: full (unfiltered, noisy) synthetic chains ===")
    model_full = train_model(full_ids, tok, args.size, args.steps, args.seed)
    print("\n=== training arm B: filtered (pruned, denser) synthetic chains ===")
    model_filtered = train_model(filtered_ids, tok, args.size, args.steps, args.seed)

    print("\n=== held-out generalization eval (shared held-out set, drawn from filtered/meaningful chains) ===")
    a_correct, a_total, a_margin = held_out_eval(model_full, tok, held_out_chains, all_fn_names)
    print(f"arm A (full, {len(full_train_chains)} chains):     {a_correct}/{a_total} correct, mean margin={a_margin:.4f}")
    b_correct, b_total, b_margin = held_out_eval(model_filtered, tok, held_out_chains, all_fn_names)
    print(f"arm B (filtered, {len(filtered_train_chains)} chains): {b_correct}/{b_total} correct, mean margin={b_margin:.4f}")


if __name__ == "__main__":
    main()
