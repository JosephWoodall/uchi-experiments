"""swarm.md Tests 1 & 2: does MoE routing collapse, and do experts route
differently for code vs. text? Reuses this session's existing joint-vocab
loader and MoE layer -- trains one model that sees both rj and code (not
two separate models), then inspects routing on held-out batches from each
domain alone.
"""
import numpy as np
import torch

from data import UNIFIED_VOCAB_SIZE, get_joint_batch, load_joint_modalities
from model import GPTConfig, TinyGPT
from tokenizer import Tokenizer

BLOCK_SIZE = 128
STEPS = 700
BATCH_SIZE = 32


def js_divergence(p, q):
    p, q = np.asarray(p) + 1e-12, np.asarray(q) + 1e-12
    p, q = p / p.sum(), q / q.sum()
    m = (p + q) / 2
    kl = lambda a, b: (a * np.log(a / b)).sum()
    return 0.5 * kl(p, m) + 0.5 * kl(q, m)


def main():
    tok = Tokenizer()
    joint = load_joint_modalities(tok)
    train_dict = {name: ids for name, (ids, _) in joint.items() if name in ("rj", "code")}
    val_dict = {name: ids for name, (_, ids) in joint.items() if name in ("rj", "code")}
    weights = {"rj": 0.5, "code": 0.5}

    cfg = GPTConfig(vocab_size=UNIFIED_VOCAB_SIZE, block_size=BLOCK_SIZE, d_model=128, n_layer=4, n_head=4,
                    moe_experts=4, moe_top_k=1)
    model = TinyGPT(cfg)
    print(f"model: {model.num_params():,} params, {cfg.moe_experts} experts")

    opt = torch.optim.AdamW(model.parameters(), lr=3e-4)
    import torch.nn.functional as F

    for step in range(1, STEPS + 1):
        model.train()
        opt.zero_grad()
        x, targets = get_joint_batch(train_dict, weights, BATCH_SIZE, BLOCK_SIZE)
        logits, _, aux_loss, _ = model(x)
        loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), targets[:, :, 0].reshape(-1)) + 0.01 * aux_loss
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        if step % 175 == 0 or step == STEPS:
            print(f"step {step}: loss={loss.item():.3f}")

    # Test 1 & 2, per block -- layer 0 alone could easily miss specialization
    # that only emerges in deeper, more context-dependent representations.
    model.eval()
    n_blocks = len(model.blocks)
    per_block_dist = {name: [[] for _ in range(n_blocks)] for name in ("rj", "code")}
    with torch.no_grad():
        for name in ("rj", "code"):
            for _ in range(10):
                x, _ = get_joint_batch({name: val_dict[name]}, {name: 1.0}, BATCH_SIZE, BLOCK_SIZE)
                model(x)
                for i, block in enumerate(model.blocks):
                    per_block_dist[name][i].append(block.mlp.last_router_probs.mean(dim=0).numpy())

    for i in range(n_blocks):
        rj_dist = np.mean(per_block_dist["rj"][i], axis=0)
        code_dist = np.mean(per_block_dist["code"][i], axis=0)
        overall = (rj_dist + code_dist) / 2
        js = js_divergence(rj_dist, code_dist)
        print(f"\n=== Block {i} ===")
        print(f"  utilization: {[f'{p:.1%}' for p in overall]}  "
              f"({'PASS' if overall.min() > 0.10 and overall.max() < 0.50 else 'FAIL'} on collapse)")
        print(f"  rj routing:   {rj_dist.round(3)}")
        print(f"  code routing: {code_dist.round(3)}")
        print(f"  JS divergence: {js:.4f}  ({'PASS' if js > 0.2 else 'FAIL'} on specialization, >0.2 bar)")


if __name__ == "__main__":
    main()
