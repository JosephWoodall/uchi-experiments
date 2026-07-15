"""Minimal GPT-style decoder, deliberately not uchi's SSM/BitNet stack —
the ablation here is about the training objective (base / mtp / jepa-aux),
not the backbone, and a plain transformer is the fastest thing to implement
correctly and the standard choice in the scaling-law literature (Kaplan et
al. 2020, Hoffmann et al. 2022). Swapping the backbone is a separate,
later question (see tasks/todo.md backlog).
"""
import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


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


class WeightQuantizer(torch.autograd.Function):
    """1.58-bit (ternary {-1,0,1}) weight quantization, straight-through
    estimator. Ported from uchi/uchi/flux/bitnet.py -- exists there to
    shrink FLUX's weight storage 20x; here it's applied to MoE experts so
    total capacity (many experts) doesn't cost proportionally more storage.
    """

    @staticmethod
    def forward(ctx, weight):
        gamma = weight.abs().mean()
        quantized = torch.round(weight / (gamma + 1e-8))
        quantized = torch.clamp(quantized, -1.0, 1.0)
        return quantized * gamma

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output


class ActivationQuantizer(torch.autograd.Function):
    """8-bit activation quantization, straight-through estimator. Ported
    alongside WeightQuantizer from uchi/uchi/flux/bitnet.py.
    """

    @staticmethod
    def forward(ctx, x):
        gamma = x.abs().max(dim=-1, keepdim=True).values
        scale = 127.0 / (gamma + 1e-8)
        quantized = torch.round(x * scale)
        quantized = torch.clamp(quantized, -128.0, 127.0)
        return quantized / scale

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output


class BitLinear(nn.Linear):
    """Ternary-weight, 8-bit-activation linear layer. Ported from uchi's
    BitLinear (uchi/uchi/flux/bitnet.py) unchanged."""

    def __init__(self, in_features, out_features, bias=False):
        super().__init__(in_features, out_features, bias)
        self.layer_norm = nn.LayerNorm(in_features)
        self.quantize = True

    def forward(self, x):
        x_norm = self.layer_norm(x)
        if self.quantize:
            x_quant = ActivationQuantizer.apply(x_norm)
            w_quant = WeightQuantizer.apply(self.weight)
            return F.linear(x_quant, w_quant, self.bias)
        return F.linear(x_norm, self.weight, self.bias)


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
    def __init__(self, cfg: GPTConfig):
        super().__init__()
        self.ln1 = nn.LayerNorm(cfg.d_model)
        self.qkv = nn.Linear(cfg.d_model, 3 * cfg.d_model)
        self.attn_out = nn.Linear(cfg.d_model, cfg.d_model)
        self.ln2 = nn.LayerNorm(cfg.d_model)
        self.is_moe = cfg.moe_experts > 0
        if self.is_moe:
            self.mlp = MoEFFN(cfg)
        else:
            self.mlp = nn.Sequential(
                nn.Linear(cfg.d_model, 4 * cfg.d_model),
                nn.GELU(),
                nn.Linear(4 * cfg.d_model, cfg.d_model),
            )
        self.n_head = cfg.n_head
        self.d_model = cfg.d_model

    def forward(self, x):
        B, T, C = x.shape
        h = self.ln1(x)
        qkv = self.qkv(h).view(B, T, 3, self.n_head, C // self.n_head).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        x = x + self.attn_out(y)
        x = x + self.mlp(self.ln2(x))
        aux_loss = self.mlp.last_aux_loss if self.is_moe else torch.tensor(0.0, device=x.device)
        return x, aux_loss


class TinyGPT(nn.Module):
    def __init__(self, cfg: GPTConfig):
        super().__init__()
        self.cfg = cfg
        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.pos_emb = nn.Embedding(cfg.block_size, cfg.d_model)
        self.blocks = nn.ModuleList([Block(cfg) for _ in range(cfg.n_layer)])
        self.ln_f = nn.LayerNorm(cfg.d_model)
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

    def hidden_states(self, idx: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """(B, T) token ids -> ((B, T, d_model) final hidden states, summed
        MoE load-balancing aux loss across layers -- 0 for dense blocks)."""
        B, T = idx.shape
        pos = torch.arange(T, device=idx.device)
        x = self.tok_emb(idx) + self.pos_emb(pos)
        aux_loss = torch.tensor(0.0, device=idx.device)
        for block in self.blocks:
            x, block_aux = block(x)
            aux_loss = aux_loss + block_aux
        return self.ln_f(x), aux_loss

    def forward(self, idx: torch.Tensor):
        """Returns (logits, extra_logits, aux_loss). extra_logits is a list
        of (B, T, vocab) tensors for the mtp arm's t+2..t+n_future heads, or
        None if n_future == 1. aux_loss is the MoE load-balancing loss
        (0 for dense models) -- callers add it to the task loss themselves.
        """
        h, aux_loss = self.hidden_states(idx)
        logits = self.lm_head(h)
        extra_logits = None
        if self.extra_heads is not None:
            extra_logits = [head(h) for head in self.extra_heads]
        return logits, extra_logits, aux_loss

    def pooled_embedding(self, idx: torch.Tensor, pad_id: int = 0) -> torch.Tensor:
        """Mean-pool non-pad hidden states, project for jepa-aux alignment."""
        assert self.proj_head is not None
        h, _ = self.hidden_states(idx)
        mask = (idx != pad_id).unsqueeze(-1).float()
        pooled = (h * mask).sum(1) / mask.sum(1).clamp(min=1)
        return self.proj_head(pooled)

    @torch.no_grad()
    def generate(self, idx: torch.Tensor, max_new_tokens: int, temperature: float = 0.8) -> torch.Tensor:
        """Autoregressive sampling for qualitative eyeballing during/after
        training — every run needs to show what it actually produces, not
        just its loss number.
        """
        self.eval()
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -self.cfg.block_size :]
            logits, _, _ = self(idx_cond)
            logits = logits[:, -1, :] / max(temperature, 1e-6)
            probs = F.softmax(logits, dim=-1)
            next_id = torch.multinomial(probs, num_samples=1)
            idx = torch.cat([idx, next_id], dim=1)
        self.train()
        return idx
