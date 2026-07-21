"""Mamba/S6-style selective variant of rwkv_model.py's TimeMixing (Gu & Dao
2023, arXiv:2312.00752): time_decay becomes a per-token, per-channel
projection of x (input-dependent), not TimeMixing's fixed, content-
independent nn.Parameter -- the actual mechanistic distinction the
architecture critique flagged (tasks/ducky.md) as untouched by the
project's five-for-five negative BPTT-retention rounds, all diagnosed on
RWKV's fixed decay specifically.

Same log-space-stable WKV recurrence as TimeMixing otherwise, same
Python-loop-plus-torch.compile approach (not a real chunked/associative
scan -- that's a separate, larger investment, flagged in tasks/todo.md,
only worth making if this first-rung check shows the mechanism is worth
it, per tasks/core_principle.md's small-scale-first rule). Isolated in
its own file, same pattern as rwkv_model.py's own header comment: testing
exactly one claim (does input-dependent decay change anything at toy
scale) in isolation from everything else.
"""
import os

import torch
import torch.nn as nn
import torch.nn.functional as F


def _selective_wkv_scan(k, v, w, u, a, b, p):
    """Same recurrence as rwkv_model.py's _wkv_scan, except w is (B, T, C)
    -- a different decay value at every timestep, not one fixed (C,) value
    shared across the whole sequence.
    """
    T = k.shape[1]
    outputs = []
    for t in range(T):
        kt, vt, wt = k[:, t, :], v[:, t, :], w[:, t, :]

        ww = u + kt
        q = torch.maximum(p, ww)
        e1 = torch.exp(p - q)
        e2 = torch.exp(ww - q)
        wkv = (e1 * a + e2 * vt) / (e1 * b + e2 + 1e-8)
        outputs.append(wkv)

        ww2 = p + wt
        q2 = torch.maximum(ww2, kt)
        e1b = torch.exp(ww2 - q2)
        e2b = torch.exp(kt - q2)
        a = e1b * a + e2b * vt
        b = e1b * b + e2b
        p = q2

    return torch.stack(outputs, dim=1), a, b, p


_USE_COMPILED_SCAN = os.environ.get("UCHI_FUSE_SSM_SCAN", "1") != "0"
try:
    _selective_scan = torch.compile(_selective_wkv_scan) if _USE_COMPILED_SCAN else _selective_wkv_scan
except Exception:
    _selective_scan = _selective_wkv_scan


class SelectiveTimeMixing(nn.Module):
    """Drop-in replacement for rwkv_model.py's TimeMixing with one change:
    decay is input-dependent (a small linear projection of x, softplus'd
    negative -- always a real decay, never growth) instead of a fixed
    learned parameter. time_first (u, the "current token" bonus) stays
    fixed, matching RWKV's own design -- only decay becomes selective,
    isolating that one variable.
    """

    def __init__(self, d_model, linear_cls=nn.Linear):
        super().__init__()
        self.time_first = nn.Parameter(torch.zeros(d_model))
        self.decay_proj = nn.Linear(d_model, d_model)
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

        w = -F.softplus(self.decay_proj(x))  # (B, T, C) -- input-dependent, always <= 0
        u = self.time_first

        if state is None:
            a = torch.zeros(B, C, device=x.device)
            b = torch.zeros(B, C, device=x.device)
            p = torch.full((B, C), -1e38, device=x.device)
        else:
            a, b, p = state

        wkv_out, a, b, p = _selective_scan(k, v, w, u, a, b, p)
        return self.output(r * wkv_out), (a, b, p)
