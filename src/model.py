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


class Block(nn.Module):
    def __init__(self, cfg: GPTConfig):
        super().__init__()
        self.ln1 = nn.LayerNorm(cfg.d_model)
        self.qkv = nn.Linear(cfg.d_model, 3 * cfg.d_model)
        self.attn_out = nn.Linear(cfg.d_model, cfg.d_model)
        self.ln2 = nn.LayerNorm(cfg.d_model)
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
        return x


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

    def hidden_states(self, idx: torch.Tensor) -> torch.Tensor:
        """(B, T) token ids -> (B, T, d_model) final hidden states."""
        B, T = idx.shape
        pos = torch.arange(T, device=idx.device)
        x = self.tok_emb(idx) + self.pos_emb(pos)
        for block in self.blocks:
            x = block(x)
        return self.ln_f(x)

    def forward(self, idx: torch.Tensor):
        """Returns (logits, extra_logits) where extra_logits is a list of
        (B, T, vocab) tensors for the mtp arm's t+2..t+n_future heads, or
        None if n_future == 1.
        """
        h = self.hidden_states(idx)
        logits = self.lm_head(h)
        extra_logits = None
        if self.extra_heads is not None:
            extra_logits = [head(h) for head in self.extra_heads]
        return logits, extra_logits

    def pooled_embedding(self, idx: torch.Tensor, pad_id: int = 0) -> torch.Tensor:
        """Mean-pool non-pad hidden states, project for jepa-aux alignment."""
        assert self.proj_head is not None
        h = self.hidden_states(idx)
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
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :] / max(temperature, 1e-6)
            probs = F.softmax(logits, dim=-1)
            next_id = torch.multinomial(probs, num_samples=1)
            idx = torch.cat([idx, next_id], dim=1)
        self.train()
        return idx
