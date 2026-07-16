"""Does training on synthetic 2-hop statements (synthetic_relations.py)
actually teach compositional generalization, or just memorize the exact
statements shown? Held-out test: split real chains into train/held-out,
synthesize training text ONLY for the train split, then check whether the
model assigns higher probability to the TRUE completion than a wrong one
for HELD-OUT chains it never saw a synthetic statement for -- if training
on other chains generalizes, this should show a real gap over a baseline
model trained with no synthetic examples at all.
"""
import argparse
import random

import torch
import torch.nn.functional as F

from data import load_lm_corpus
from model import GPTConfig, TinyGPT
from synthetic_relations import build_call_graph, find_transitive_chains, synthesize_examples
from tokenizer import Tokenizer
from train import SIZES

EVAL_TEMPLATE = "# {a} -> {b} -> "


def held_out_eval(model, tok, held_out_chains, all_fn_names, seed=0):
    """For each held-out chain (a, b, c), compare the model's average
    per-token log-prob of the TRUE continuation "c" against a WRONG
    distractor function name (real, just not b's actual callee) --
    teacher-forced, same mechanism as grounding.py's self_critique_score.
    Returns (n_correct, n_total, mean_margin) -- correct = true scores
    higher than wrong.
    """
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
    p.add_argument("--steps", type=int, default=1500)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--num-threads", type=int, default=8)
    args = p.parse_args()
    torch.set_num_threads(args.num_threads)

    tok = Tokenizer()
    from pathlib import Path
    ROOT = Path(__file__).resolve().parent.parent
    code_source = (ROOT / "data" / "code" / "corpus.txt").read_text()

    call_graph = build_call_graph(code_source)
    all_fn_names = sorted(call_graph.keys())
    chains = find_transitive_chains(call_graph)
    rng = random.Random(42)
    rng.shuffle(chains)
    n_held_out = max(1, len(chains) // 10)
    held_out_chains = chains[:n_held_out]
    train_chains = chains[n_held_out:]
    print(f"total chains={len(chains)}  train={len(train_chains)}  held_out={len(held_out_chains)}")

    train_examples = synthesize_examples(train_chains, seed=0)
    synthetic_text = "\n".join(train_examples)

    code_train_ids, _ = load_lm_corpus("code", tok)
    synthetic_ids = torch.tensor(tok.encode(synthetic_text), dtype=torch.long)

    baseline_ids = code_train_ids
    treatment_ids = torch.cat([code_train_ids, synthetic_ids])
    print(f"baseline training tokens={len(baseline_ids)}  "
          f"treatment (real + synthetic) tokens={len(treatment_ids)}  "
          f"synthetic tokens added={len(synthetic_ids)}")

    print("\n=== training baseline (no synthetic examples) ===")
    baseline_model = train_model(baseline_ids, tok, args.size, args.steps, args.seed)
    print("\n=== training treatment (+ synthetic 2-hop statements, held-out chains excluded) ===")
    treatment_model = train_model(treatment_ids, tok, args.size, args.steps, args.seed)

    print("\n=== held-out generalization eval (chains NEVER shown as synthetic statements) ===")
    b_correct, b_total, b_margin = held_out_eval(baseline_model, tok, held_out_chains, all_fn_names)
    print(f"baseline:  {b_correct}/{b_total} correct (true > wrong), mean margin={b_margin:.4f}")
    t_correct, t_total, t_margin = held_out_eval(treatment_model, tok, held_out_chains, all_fn_names)
    print(f"treatment: {t_correct}/{t_total} correct (true > wrong), mean margin={t_margin:.4f}")


if __name__ == "__main__":
    main()
