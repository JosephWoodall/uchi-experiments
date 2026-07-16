"""MCTS-lite: value-guided search over generation branches, modeled on
uchi's mcts.py (PUCT/UCB/backprop mechanics, hypothesis-level actions
rather than per-token search) but adapted to what Ducky actually has.

Two honest differences from uchi's version, stated plainly rather than
glossed over:
1. uchi's MCTS scores leaves with a WorldModel.ValueHead -- a trained
   neural network mapping hidden state -> expected value. Ducky has no
   such thing (training one needs real reward signal from a proposer/
   verifier loop uchi's own docs say is still gated). This uses
   self_critique_score (+ syntax-validity for code) as a cheap, already-
   validated proxy instead -- not a trained value function, an honest
   stand-in, same "no second model" discipline as reject-and-resample.
2. uchi approximates depth>1 rollout with a neutral action probe through
   a learned dynamics model, because sampling real continuations at every
   simulated tree depth would be too expensive at their scale. At Ducky's
   toy scale, real sampling at depth is cheap enough to just do directly --
   so every simulated node here is a REAL generated continuation, not an
   imagined one. Strictly more honest than the approximation it's modeled
   on, not a compromise.

Branching happens at chunk granularity (chunk_size tokens per action),
matching uchi's "one action is a whole Thought/Action continuation, not a
single vocab token" redefinition -- adapted to Ducky's code/rj domains as
"one action is a chunk of continued generation," the natural granularity
here since there's no Thought/Action structure to split on.
"""
import math

import torch

from grounding import self_critique_score, verify_code_syntax
from inference import ABSTAIN, predict_next


class _MCTSNode:
    __slots__ = ("tokens", "prior", "parent", "children", "visit_count", "value_sum", "done")

    def __init__(self, tokens: list, prior: float = 1.0, parent=None):
        self.tokens = tokens  # full generated-so-far token list (prompt excluded)
        self.prior = prior
        self.parent = parent
        self.children: list = []
        self.visit_count = 0
        self.value_sum = 0.0
        self.done = False  # hit ABSTAIN or max_new_tokens -- no further expansion possible

    @property
    def value(self) -> float:
        return self.value_sum / max(self.visit_count, 1)

    def ucb(self, c_puct: float = 1.5) -> float:
        if self.parent is None:
            return 0.0
        exploration = c_puct * self.prior * math.sqrt(self.parent.visit_count) / (1 + self.visit_count)
        return self.value + exploration

    def is_leaf(self) -> bool:
        return len(self.children) == 0


def mcts_generate(model, tok, graph, prompt: str, max_new_tokens: int, domain: str,
                   fast_threshold: float, abstain_threshold: float, slow_abstain_threshold: float,
                   symbol_table: set = None, ngram_index: set = None, ngram_n: int = 4,
                   chunk_size: int = 8, top_k: int = 3, n_simulations: int = 6,
                   c_puct: float = 1.5, temperature: float = 0.8):
    """Value-guided search over chunk-level generation branches. Returns
    the same result-dict shape as generate_with_grounding/_resampling, plus
    mcts_simulations/mcts_final_depth_tokens for observability into what
    the search actually did.
    """
    prompt_ids = tok.encode(prompt)
    root = _MCTSNode(tokens=[])

    def full_ids(node: "_MCTSNode") -> torch.Tensor:
        return torch.tensor([prompt_ids + node.tokens], dtype=torch.long)

    def value_of(node: "_MCTSNode") -> float:
        if not node.tokens:
            return 0.0
        prompt_ids_t = torch.tensor([prompt_ids], dtype=torch.long)
        score = self_critique_score(model, prompt_ids_t, node.tokens)
        if domain == "code":
            text = tok.decode(node.tokens)
            if verify_code_syntax(prompt + text):
                score += 1.0  # same decisive syntax bonus as reject-and-resample's scoring
        return score

    def expand(node: "_MCTSNode") -> None:
        if node.done or len(node.tokens) >= max_new_tokens:
            node.done = True
            return
        raw_children = []
        for _ in range(top_k):
            ids = full_ids(node)
            chunk_tokens: list = []
            log_probs: list = []
            budget = min(chunk_size, max_new_tokens - len(node.tokens))
            for _ in range(budget):
                next_id, info = predict_next(model, graph, ids, fast_threshold, abstain_threshold,
                                              slow_abstain_threshold, ngram_index=ngram_index,
                                              ngram_n=ngram_n, temperature=temperature)
                if next_id == ABSTAIN:
                    break
                chunk_tokens.append(next_id)
                log_probs.append(math.log(max(info.get("confidence", 1e-6), 1e-6)))
                ids = torch.cat([ids, torch.tensor([[next_id]])], dim=1)
            if not chunk_tokens:
                continue  # this candidate abstained immediately -- not a viable branch
            avg_log_prob = sum(log_probs) / len(log_probs)
            raw_children.append((chunk_tokens, avg_log_prob))

        if not raw_children:
            node.done = True
            return
        lps = torch.tensor([lp for _, lp in raw_children])
        priors = torch.softmax(lps, dim=0).tolist()
        for (chunk_tokens, _), p in zip(raw_children, priors):
            node.children.append(_MCTSNode(tokens=node.tokens + chunk_tokens, prior=p, parent=node))

    def select(node: "_MCTSNode") -> "_MCTSNode":
        while not node.is_leaf():
            node = max(node.children, key=lambda n: n.ucb(c_puct))
        return node

    def backprop(node, value: float) -> None:
        while node is not None:
            node.visit_count += 1
            node.value_sum += value
            node = node.parent

    for _ in range(n_simulations):
        leaf = select(root)
        if leaf.visit_count > 0 and not leaf.done and leaf.is_leaf():
            expand(leaf)
            if leaf.children:
                leaf = select(leaf)
        value = value_of(leaf)
        backprop(leaf, value)

    # Robust final selection: most-visited child at each level, not
    # highest single-rollout value -- standard MCTS action policy, avoids
    # being fooled by one lucky simulation.
    node = root
    while node.children:
        node = max(node.children, key=lambda n: n.visit_count)

    text = tok.decode(node.tokens)
    result = {"prompt": prompt, "generated_text": text, "n_tokens_generated": len(node.tokens),
              "abstained_at_token": None if node.tokens else 0,
              "mcts_simulations": n_simulations, "mcts_final_depth_tokens": len(node.tokens)}
    if node.tokens:
        result["self_critique_score"] = self_critique_score(
            model, torch.tensor([prompt_ids], dtype=torch.long), node.tokens)
        if domain == "code":
            full_text = prompt + text
            result["syntax_valid"] = verify_code_syntax(full_text)
            if symbol_table is not None:
                from grounding import identifier_grounded
                result["identifier_grounded"] = identifier_grounded(full_text, symbol_table)
    return result
