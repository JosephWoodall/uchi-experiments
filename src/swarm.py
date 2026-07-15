"""Swarm inference for Tests 4 & 5 (swarm.md): 3 experts, each combining the
SAME trained neural model's logits with a different graph-query heuristic --
not the original 6 exotic strategies (local/long-range/frequency/fact-
grounded/syntax/novelty), cut down per tasks/todo.md Phase F. Reuses an
already-trained checkpoint rather than training a new model, since the
question here is about the inference-time mechanism, not training.
"""
import json
from pathlib import Path

import torch
import torch.nn.functional as F

from graph import build_graph
from model import GPTConfig, TinyGPT
from tokenizer import Tokenizer
from train import SIZES

ROOT = Path(__file__).resolve().parent.parent


def query_local(graph, last_token, vocab_size):
    scores = torch.zeros(vocab_size)
    for tgt, meta in graph.successors(last_token).items():
        scores[tgt] = meta["weight"] * meta["confidence"]
    return scores


def query_frequency(graph, last_token, vocab_size):
    scores = torch.zeros(vocab_size)
    for tgt, meta in graph.successors(last_token).items():
        if meta["relation_type"] == "co_occurrence":
            scores[tgt] = meta.get("frequency", 0) * meta["confidence"]
        else:
            scores[tgt] = meta["weight"] * meta["confidence"]
    return scores


def query_facts_only(graph, last_token, vocab_size):
    scores = torch.zeros(vocab_size)
    for tgt, meta in graph.successors(last_token).items():
        if meta["relation_type"] == "semantic_fact":
            scores[tgt] = meta["confidence"]
    return scores


HEURISTICS = {"local": query_local, "frequency": query_frequency, "facts": query_facts_only}


def load_checkpoint(run_name, checkpoint="model_best.pt"):
    run_dir = ROOT / "runs" / run_name
    cfg_dict = json.loads((run_dir / "config.json").read_text())
    tok = Tokenizer()
    size_cfg = SIZES[cfg_dict["size"]]
    cfg = GPTConfig(vocab_size=tok.vocab_size, block_size=cfg_dict["block_size"], **size_cfg)
    model = TinyGPT(cfg)
    model.load_state_dict(torch.load(run_dir / checkpoint, map_location="cpu"))
    model.eval()
    return model, tok


@torch.no_grad()
def generate(model, graph, prompt_ids, max_new_tokens, use_graph, use_swarm,
             alpha=0.6, beta=0.4, temperature=0.8):
    ids = prompt_ids.clone()
    for _ in range(max_new_tokens):
        idx_cond = ids[:, -model.cfg.block_size:]
        logits, _, _, _ = model(idx_cond)
        neural_logits = logits[:, -1, :]
        last_token = ids[0, -1].item()

        if not use_graph:
            combined = neural_logits
        else:
            names = list(HEURISTICS.keys()) if use_swarm else ["local"]
            expert_logits, expert_weights = [], []
            for name in names:
                graph_scores = HEURISTICS[name](graph, last_token, neural_logits.size(-1)).unsqueeze(0)
                combined_logits = alpha * neural_logits + beta * graph_scores
                conf = F.softmax(combined_logits, dim=-1).max().item()
                expert_logits.append(combined_logits)
                expert_weights.append(conf)
            w = torch.tensor(expert_weights)
            w = w / w.sum()
            combined = sum(wi * lg for wi, lg in zip(w, expert_logits))

        probs = F.softmax(combined / temperature, dim=-1)
        next_id = torch.multinomial(probs, num_samples=1)
        ids = torch.cat([ids, next_id], dim=1)
    return ids[0].tolist()


def token_diff(a, b):
    n = min(len(a), len(b))
    return sum(1 for i in range(n) if a[i] != b[i]) / n


def main():
    model, tok = load_checkpoint("rj_base_m")
    code_source = (ROOT / "data" / "code" / "corpus.txt").read_text()
    rj_ids = torch.load(ROOT / "data" / "cache" / "rj.pt")
    code_ids = torch.load(ROOT / "data" / "cache" / "code.pt")
    graph = build_graph(tok, code_source, rj_ids, code_ids)

    prompt = torch.tensor([tok.encode("ROMEO:")], dtype=torch.long)
    n_new = 60

    torch.manual_seed(1)
    no_graph = generate(model, graph, prompt, n_new, use_graph=False, use_swarm=False)
    torch.manual_seed(1)
    single_with_graph = generate(model, graph, prompt, n_new, use_graph=True, use_swarm=False)
    torch.manual_seed(1)
    swarm_with_graph = generate(model, graph, prompt, n_new, use_graph=True, use_swarm=True)

    print("no graph (pure neural):      ", tok.decode(no_graph))
    print("single expert + graph:       ", tok.decode(single_with_graph))
    print("swarm (3 heuristics) + graph:", tok.decode(swarm_with_graph))

    diff_graph = token_diff(no_graph, single_with_graph)
    diff_swarm = token_diff(single_with_graph, swarm_with_graph)

    print("\n=== Test 5: graph affects predictions ===")
    print(f"token diff (no graph vs single+graph): {diff_graph:.1%}")
    print("PASS (>10%)" if diff_graph > 0.10 else "FAIL (<2% would mean graph integration broken)")

    print("\n=== Test 4: swarm differs from single expert ===")
    print(f"token diff (single+graph vs swarm+graph): {diff_swarm:.1%}")
    print("PASS (>20%)" if diff_swarm > 0.20 else "FAIL (<5% would mean swarm is redundant)")


if __name__ == "__main__":
    main()
