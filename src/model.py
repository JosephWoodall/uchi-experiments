"""Minimal GPT-style decoder, deliberately not uchi's SSM/BitNet stack —
the ablation here is about the training objective (base / mtp / jepa-aux),
not the backbone, and a plain transformer is the fastest thing to implement
correctly and the standard choice in the scaling-law literature (Kaplan et
al. 2020, Hoffmann et al. 2022). Swapping the backbone is a separate,
later question (see tasks/todo.md backlog).
"""
import math
from dataclasses import dataclass, field

import torch
import torch.nn as nn
import torch.nn.functional as F

from bitnet import BitLinear
from rwkv_model import TimeMixing


@dataclass
class GPTConfig:
    vocab_size: int
    block_size: int
    d_model: int
    n_layer: int
    n_head: int
    n_future: int = 1  # 1 = base/jepa-aux, >1 = mtp
    proj_dim: int = 0  # >0 = jepa-aux enabled, adds a projection head
    moe_experts: int = 0  # 0 = dense MLP; >0 = MoE FFN with this many routed experts
    moe_top_k: int = 1
    use_bitlinear_experts: bool = False  # ported from uchi/uchi/flux/bitnet.py
    use_bitlinear: bool = False  # BitLinear throughout Ducky's own blocks (attention/RWKV
    # projections + dense MLP), not just the (abandoned) MoE experts above
    embedding_rank: int = 0  # 0 = plain tied nn.Embedding (unchanged); >0 = TensorRankEmbedding
    # (uchi-style low-rank factored embedding) at this rank, output head reuses the same
    # factorization transposed (uchi's "symmetric factored projection" for the output head --
    # not uchi's separate syntax-prediction DualHead, which needs labels we don't have)
    use_rwkv_hybrid: bool = False  # mostly RWKV time-mixing blocks (unlimited context, O(1)
    # memory), periodic attention for in-window quality -- mirrors uchi's own SSM +
    # periodic-attention design (uchi/README.md). Ignored if False (pure attention, unchanged).
    attention_layers: tuple = field(default_factory=tuple)  # 0-indexed layers that use attention
    # when use_rwkv_hybrid=True; all others use RWKV time-mixing.


class Expert(nn.Module):
    def __init__(self, d_model: int, d_hidden: int, use_bitlinear: bool = False):
        super().__init__()
        Linear = BitLinear if use_bitlinear else nn.Linear
        self.fc1 = Linear(d_model, d_hidden)
        self.fc2 = Linear(d_hidden, d_model)

    def forward(self, x):
        return self.fc2(F.gelu(self.fc1(x)))


class MoEFFN(nn.Module):
    """Top-k routed experts + one always-on shared expert (DeepSeekMoE-style
    fine-grained + shared expert, arXiv:2401.06066), optionally BitLinear-
    quantized. Expert hidden size is 2*d_model (vs. dense's 4*d_model) so
    that active params per token (shared + top_1) exactly match the dense
    block's MLP param count -- same active compute, more total capacity via
    the unused routed experts.
    """

    def __init__(self, cfg: "GPTConfig"):
        super().__init__()
        d_model = cfg.d_model
        d_hidden = 2 * d_model
        self.n_experts = cfg.moe_experts
        self.top_k = cfg.moe_top_k
        self.router = nn.Linear(d_model, self.n_experts, bias=False)
        self.experts = nn.ModuleList(
            [Expert(d_model, d_hidden, cfg.use_bitlinear_experts) for _ in range(self.n_experts)]
        )
        self.shared_expert = Expert(d_model, d_hidden, cfg.use_bitlinear_experts)
        self.last_aux_loss = torch.tensor(0.0)

    def forward(self, x):
        B, T, C = x.shape
        flat = x.reshape(-1, C)
        router_logits = self.router(flat)
        router_probs = F.softmax(router_logits, dim=-1)
        topk_probs, topk_idx = router_probs.topk(self.top_k, dim=-1)
        topk_probs = topk_probs / topk_probs.sum(dim=-1, keepdim=True)

        out = torch.zeros_like(flat)
        for slot in range(self.top_k):
            idx = topk_idx[:, slot]
            weight = topk_probs[:, slot]
            for e in range(self.n_experts):
                mask = idx == e
                if mask.any():
                    out[mask] = out[mask] + weight[mask].unsqueeze(-1) * self.experts[e](flat[mask])
        out = out + self.shared_expert(flat)

        # Switch-Transformer load-balancing aux loss: penalize uneven
        # routing so experts don't collapse to using just one or two.
        frac_routed = torch.stack([(topk_idx == e).float().mean() for e in range(self.n_experts)])
        avg_prob = router_probs.mean(dim=0)
        self.last_aux_loss = self.n_experts * (frac_routed * avg_prob).sum()

        # Exposed for analysis (expert utilization, per-modality routing
        # divergence -- swarm.md Tests 1 & 2), not used in the loss itself.
        self.last_router_probs = router_probs.detach()
        self.last_topk_idx = topk_idx.detach()

        return out.reshape(B, T, C)


class Block(nn.Module):
    def __init__(self, cfg: GPTConfig, layer_idx: int = 0):
        super().__init__()
        Linear = BitLinear if cfg.use_bitlinear else nn.Linear
        self.ln1 = nn.LayerNorm(cfg.d_model)
        self.is_attention = (not cfg.use_rwkv_hybrid) or (layer_idx in cfg.attention_layers)
        if self.is_attention:
            self.qkv = Linear(cfg.d_model, 3 * cfg.d_model)
            self.attn_out = Linear(cfg.d_model, cfg.d_model)
        else:
            # RWKV time-mixing: gated linear recurrence, O(1) state per
            # channel regardless of sequence length (see rwkv_model.py).
            # Unlike the attention path above, this can carry state across
            # chunks far longer than block_size -- that's the whole point.
            self.time_mixing = TimeMixing(cfg.d_model, linear_cls=Linear)
        self.ln2 = nn.LayerNorm(cfg.d_model)
        self.is_moe = cfg.moe_experts > 0
        if self.is_moe:
            self.mlp = MoEFFN(cfg)
        else:
            self.mlp = nn.Sequential(
                Linear(cfg.d_model, 4 * cfg.d_model),
                nn.GELU(),
                Linear(4 * cfg.d_model, cfg.d_model),
            )
        self.n_head = cfg.n_head
        self.d_model = cfg.d_model

    def forward(self, x, rwkv_state=None):
        if self.is_attention:
            B, T, C = x.shape
            h = self.ln1(x)
            qkv = self.qkv(h).view(B, T, 3, self.n_head, C // self.n_head).permute(2, 0, 3, 1, 4)
            q, k, v = qkv[0], qkv[1], qkv[2]
            y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
            y = y.transpose(1, 2).contiguous().view(B, T, C)
            x = x + self.attn_out(y)
            new_rwkv_state = None
        else:
            dx, new_rwkv_state = self.time_mixing(self.ln1(x), rwkv_state)
            x = x + dx
        x = x + self.mlp(self.ln2(x))
        aux_loss = self.mlp.last_aux_loss if self.is_moe else torch.tensor(0.0, device=x.device)
        return x, aux_loss, new_rwkv_state


class TensorRankEmbedding(nn.Module):
    """Low-rank factored embedding (uchi-style Tucker decomposition):
    embed(i) = W2 @ W1 @ e_i, rank r << d_model. Cost: V*r + r*d vs V*d for
    a full table -- at our 1024-token vocab this is still a real cut (e.g.
    r=32, d=128: ~36.9K vs 131K params, ~72% reduction), not negligible
    just because the vocab is small. The output head reuses the SAME
    factorization transposed (uchi's own "symmetric factored projection
    for the output head") -- this is the parameter-efficiency mechanism,
    not uchi's separate syntax-prediction DualHead, which needs labeled
    syntax tokens we don't have.
    """

    def __init__(self, vocab_size: int, d_model: int, rank: int):
        super().__init__()
        self.w1 = nn.Parameter(torch.randn(vocab_size, rank) * 0.02)
        self.w2 = nn.Parameter(torch.randn(rank, d_model) * 0.02)

    def embed(self, idx: torch.Tensor) -> torch.Tensor:
        return F.embedding(idx, self.w1) @ self.w2

    def project(self, h: torch.Tensor) -> torch.Tensor:
        return h @ self.w2.T @ self.w1.T


class TinyGPT(nn.Module):
    def __init__(self, cfg: GPTConfig):
        super().__init__()
        self.cfg = cfg
        self.use_factored_embedding = cfg.embedding_rank > 0
        if self.use_factored_embedding:
            self.factored_emb = TensorRankEmbedding(cfg.vocab_size, cfg.d_model, cfg.embedding_rank)
        else:
            self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.pos_emb = nn.Embedding(cfg.block_size, cfg.d_model)
        self.blocks = nn.ModuleList([Block(cfg, layer_idx=i) for i in range(cfg.n_layer)])
        self.ln_f = nn.LayerNorm(cfg.d_model)
        if not self.use_factored_embedding:
            self.lm_head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
            self.lm_head.weight = self.tok_emb.weight  # tied, standard practice

        self.extra_heads = None
        if cfg.n_future > 1:
            self.extra_heads = nn.ModuleList(
                [nn.Linear(cfg.d_model, cfg.vocab_size) for _ in range(cfg.n_future - 1)]
            )

        self.proj_head = None
        if cfg.proj_dim > 0:
            self.proj_head = nn.Linear(cfg.d_model, cfg.proj_dim)

        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(m):
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)

    def num_params(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def hidden_states(self, idx: torch.Tensor, rwkv_states=None):
        """(B, T) token ids -> ((B, T, d_model) final hidden states, summed
        MoE load-balancing aux loss across layers -- 0 for dense blocks,
        new per-block RWKV states -- None entries for attention blocks).
        rwkv_states, if given, is a list of per-block states to continue a
        recurrence across chunks (the unlimited-context mechanism); None
        starts fresh, matching ordinary training/short-context use.
        """
        B, T = idx.shape
        pos = torch.arange(T, device=idx.device)
        tok_x = self.factored_emb.embed(idx) if self.use_factored_embedding else self.tok_emb(idx)
        x = tok_x + self.pos_emb(pos)
        aux_loss = torch.tensor(0.0, device=idx.device)
        new_states = []
        for i, block in enumerate(self.blocks):
            state = rwkv_states[i] if rwkv_states is not None else None
            x, block_aux, new_state = block(x, state)
            aux_loss = aux_loss + block_aux
            new_states.append(new_state)
        return self.ln_f(x), aux_loss, new_states

    def forward(self, idx: torch.Tensor, rwkv_states=None):
        """Returns (logits, extra_logits, aux_loss, new_rwkv_states).
        extra_logits is a list of (B, T, vocab) tensors for the mtp arm's
        t+2..t+n_future heads, or None if n_future == 1. aux_loss is the MoE
        load-balancing loss (0 for dense models) -- callers add it to the
        task loss themselves. new_rwkv_states is only relevant for
        use_rwkv_hybrid models doing chunked long-context inference; every
        other caller can ignore it (state=None each call, unchanged
        short-context behavior).
        """
        h, aux_loss, new_states = self.hidden_states(idx, rwkv_states)
        logits = self.factored_emb.project(h) if self.use_factored_embedding else self.lm_head(h)
        extra_logits = None
        if self.extra_heads is not None:
            extra_logits = [head(h) for head in self.extra_heads]
        return logits, extra_logits, aux_loss, new_states

    def pooled_embedding(self, idx: torch.Tensor, pad_id: int = 0) -> torch.Tensor:
        """Mean-pool non-pad hidden states, project for jepa-aux alignment."""
        assert self.proj_head is not None
        h, _, _ = self.hidden_states(idx)
        mask = (idx != pad_id).unsqueeze(-1).float()
        pooled = (h * mask).sum(1) / mask.sum(1).clamp(min=1)
        return self.proj_head(pooled)

    @torch.no_grad()
    def generate(self, idx: torch.Tensor, max_new_tokens: int, temperature: float = 0.8) -> torch.Tensor:
        """Autoregressive sampling for qualitative eyeballing during/after
        training — every run needs to show what it actually produces, not
        just its loss number. Always fresh state each step (crops to the
        last block_size tokens) -- matches prior behavior for every
        existing (non-hybrid) config; long-context chunked generation with
        carried RWKV state is a separate, dedicated code path (see
        test_unlimited_context.py), not this one.
        """
        self.eval()
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -self.cfg.block_size :]
            logits, _, _, _ = self(idx_cond)
            logits = logits[:, -1, :] / max(temperature, 1e-6)
            probs = F.softmax(logits, dim=-1)
            next_id = torch.multinomial(probs, num_samples=1)
            idx = torch.cat([idx, next_id], dim=1)
        self.train()
        return idx
