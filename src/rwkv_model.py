"""Minimal RWKV (Peng et al. 2023, arXiv:2305.13048), isolated from
MoE/graph/swarm -- testing exactly one claim from tasks/swarm.md: does a
linear-recurrent backbone give O(1) memory per token regardless of context
length, unlike TinyGPT's hard block_size=128 cap (model.py's generate()
structurally cannot see past the last 128 tokens; this can).

Time-mixing (WKV) replaces self-attention; channel-mixing replaces the FFN.
Numerically-stable recurrence (running max in log-space, matching the real
RWKV formulation), executed as an explicit Python time-loop rather than a
fused kernel -- fine at our sequence lengths, not fine at RWKV-7B scale
(that's what the real CUDA kernel is for).

No positional embedding: the recurrence itself encodes sequence order,
unlike TinyGPT's learned pos_emb table (which is itself another source of
TinyGPT's hard length cap).
"""
import os
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class RWKVConfig:
    vocab_size: int
    d_model: int
    n_layer: int


def _wkv_scan(k, v, w, u, a, b, p):
    """Sequential WKV recurrence, numerically stable (running max in
    log-space). Extracted as a standalone function -- not a method -- so a
    single compiled version is shared across every TimeMixing layer/
    instance, rather than each one separately retracing the identical
    algorithm on identical shapes.

    Matches uchi's own fix for the identical bottleneck (uchi/README.md:
    `UCHI_FUSE_SSM_SCAN=1` fuses their sequential SSM scan via
    torch.compile over the whole loop, not per-step -- and explicitly not
    a switch to a parallel/associative scan, which they tried and reverted
    for measured memory-bandwidth reasons). Same algorithm as before,
    same math, just letting the compiler fuse the many small per-step ops
    into fewer kernel launches instead of paying Python dispatch overhead
    128 times per forward call.
    """
    T = k.shape[1]
    outputs = []
    for t in range(T):
        kt, vt = k[:, t, :], v[:, t, :]

        ww = u + kt
        q = torch.maximum(p, ww)
        e1 = torch.exp(p - q)
        e2 = torch.exp(ww - q)
        wkv = (e1 * a + e2 * vt) / (e1 * b + e2 + 1e-8)
        outputs.append(wkv)

        ww2 = p + w
        q2 = torch.maximum(ww2, kt)
        e1b = torch.exp(ww2 - q2)
        e2b = torch.exp(kt - q2)
        a = e1b * a + e2b * vt
        b = e1b * b + e2b
        p = q2

    return torch.stack(outputs, dim=1), a, b, p


# Compiled once at import time and shared across every TimeMixing instance.
# UCHI_FUSE_SSM_SCAN=0 disables it (falls back to the eager loop) in case
# torch.compile isn't happy in a given environment -- same escape hatch
# name uchi itself uses for the identical decision.
_USE_COMPILED_SCAN = os.environ.get("UCHI_FUSE_SSM_SCAN", "1") != "0"
try:
    _scan = torch.compile(_wkv_scan) if _USE_COMPILED_SCAN else _wkv_scan
except Exception:
    _scan = _wkv_scan


class TimeMixing(nn.Module):
    """WKV: gated linear recurrence. State (a, b, p) per channel is fixed
    size regardless of how many tokens have been processed -- this is the
    mechanism, not just an inference trick.
    """

    def __init__(self, d_model, linear_cls=nn.Linear):
        super().__init__()
        self.time_decay = nn.Parameter(torch.zeros(d_model))
        self.time_first = nn.Parameter(torch.zeros(d_model))
        self.key = linear_cls(d_model, d_model, bias=False)
        self.value = linear_cls(d_model, d_model, bias=False)
        self.receptance = linear_cls(d_model, d_model, bias=False)
        self.output = linear_cls(d_model, d_model, bias=False)
        self.time_mix_k = nn.Parameter(torch.ones(1, 1, d_model) * 0.5)
        self.time_mix_v = nn.Parameter(torch.ones(1, 1, d_model) * 0.5)
        self.time_mix_r = nn.Parameter(torch.ones(1, 1, d_model) * 0.5)

    @staticmethod
    def token_shift(x):
        return F.pad(x, (0, 0, 1, -1))

    def forward(self, x, state=None):
        B, T, C = x.shape
        xs = self.token_shift(x)
        xk = x * self.time_mix_k + xs * (1 - self.time_mix_k)
        xv = x * self.time_mix_v + xs * (1 - self.time_mix_v)
        xr = x * self.time_mix_r + xs * (1 - self.time_mix_r)

        k = self.key(xk)
        v = self.value(xv)
        r = torch.sigmoid(self.receptance(xr))

        w = -torch.exp(self.time_decay)
        u = self.time_first

        if state is None:
            a = torch.zeros(B, C, device=x.device)
            b = torch.zeros(B, C, device=x.device)
            p = torch.full((B, C), -1e38, device=x.device)
        else:
            a, b, p = state

        wkv_out, a, b, p = _scan(k, v, w, u, a, b, p)
        return self.output(r * wkv_out), (a, b, p)


class ChannelMixing(nn.Module):
    """RWKV's FFN replacement: token-shift + squared-ReLU, receptance-gated."""

    def __init__(self, d_model, d_hidden=None):
        super().__init__()
        d_hidden = d_hidden or 4 * d_model
        self.time_mix_k = nn.Parameter(torch.ones(1, 1, d_model) * 0.5)
        self.time_mix_r = nn.Parameter(torch.ones(1, 1, d_model) * 0.5)
        self.key = nn.Linear(d_model, d_hidden, bias=False)
        self.receptance = nn.Linear(d_model, d_model, bias=False)
        self.value = nn.Linear(d_hidden, d_model, bias=False)

    @staticmethod
    def token_shift(x):
        return F.pad(x, (0, 0, 1, -1))

    def forward(self, x):
        xs = self.token_shift(x)
        xk = x * self.time_mix_k + xs * (1 - self.time_mix_k)
        xr = x * self.time_mix_r + xs * (1 - self.time_mix_r)
        k = torch.square(torch.relu(self.key(xk)))
        v = self.value(k)
        r = torch.sigmoid(self.receptance(xr))
        return r * v


class RWKVBlock(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.ln2 = nn.LayerNorm(d_model)
        self.time_mixing = TimeMixing(d_model)
        self.channel_mixing = ChannelMixing(d_model)

    def forward(self, x, state=None):
        dx, new_state = self.time_mixing(self.ln1(x), state)
        x = x + dx
        x = x + self.channel_mixing(self.ln2(x))
        return x, new_state


class RWKVModel(nn.Module):
    def __init__(self, cfg: RWKVConfig):
        super().__init__()
        self.cfg = cfg
        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.blocks = nn.ModuleList([RWKVBlock(cfg.d_model) for _ in range(cfg.n_layer)])
        self.ln_f = nn.LayerNorm(cfg.d_model)
        self.head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
        self.head.weight = self.tok_emb.weight
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(m):
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)

    def num_params(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def forward(self, idx, states=None):
        """idx: (B, T). states: optional list of per-block (a,b,p) states to
        continue a recurrence across chunks -- process chunk 2 without ever
        re-seeing chunk 1's tokens, carrying forward only the fixed-size
        state. This is the actual unlimited-context mechanism, not generate()
        truncating context the way TinyGPT must.
        """
        x = self.tok_emb(idx)
        new_states = []
        for i, block in enumerate(self.blocks):
            state = states[i] if states is not None else None
            x, new_state = block(x, state)
            new_states.append(new_state)
        x = self.ln_f(x)
        return self.head(x), new_states
