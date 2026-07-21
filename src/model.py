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
from mamba_lite import SelectiveTimeMixing
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
    tie_layers: bool = False  # Universal-Transformer/ALBERT-style: reuse ONE physical block's
    # weights at every depth step instead of n_layer independent blocks -- tests whether "more
    # effective passes through fewer parameter sets" beats "more independent layers" at a cut
    # param budget (see tasks/ducky.md's architecture critique, ranked idea #2). Requires a
    # homogeneous block type: a single shared block can't be RWKV at one depth and attention at
    # another, so this is only valid with attention_layers=() (pure RWKV throughout, if
    # use_rwkv_hybrid) or use_rwkv_hybrid=False (pure attention, unchanged either way).
    use_halting: bool = False  # per-block halting head (Linear(d_model, 1), negligible params),
    # trained via an auxiliary BCE loss predicting "does THIS layer's own logit-lens prediction
    # already match the final-depth prediction" -- the same signal eval_early_exit.py's untrained
    # probe checked empirically, now actually learned instead of borrowed (see
    # tasks/ducky.md's architecture critique, ranked idea #1, and TinyGPT.halting_loss below).
    use_selective_decay: bool = False  # mamba_lite.py's SelectiveTimeMixing instead of
    # rwkv_model.py's TimeMixing for non-attention blocks when use_rwkv_hybrid=True -- input-
    # dependent decay (Mamba/S6-style, Gu & Dao 2023) instead of RWKV's fixed per-channel decay,
    # the actual mechanistic distinction from RWKV flagged in the architecture critique (ranked
    # idea #3). Ignored unless use_rwkv_hybrid=True.
    selective_decay_layers: tuple = field(default_factory=tuple)  # 0-indexed non-attention layers
    # that use SelectiveTimeMixing specifically -- more precise than use_selective_decay's
    # all-or-nothing (see tasks/ducky.md: uniform selective decay lost to plain RWKV on all 3
    # domains in the scaling sweep; this tests restricting it to attention-adjacent layers only).
    # Non-empty overrides use_selective_decay entirely; empty (default) falls back to that
    # boolean's exact existing all-or-nothing behavior -- fully backward compatible.
    use_width_gating: bool = False  # WidthGatedMLP instead of the plain dense MLP -- the
    # width-axis analog of use_halting's depth-axis question: "how much of this block's FFN
    # capacity does this token need," not "how many blocks." See WidthGatedMLP and
    # TinyGPT.width_sparsity_loss below. Ignored when moe_experts > 0.
    scaled_residual_init: bool = False  # GPT-2-paper residual-projection init (Radford et al.
    # 2019; ported via nanoGPT): the last linear in each residual branch (attention's attn_out,
    # dense/width-gated MLP's second projection) gets std=0.02/sqrt(2*n_layer) instead of the
    # flat 0.02 everything else uses, to stop the residual stream's variance from growing with
    # depth. Neither this file nor uchi's own flux/model.py had this until now -- part of
    # --nanogpt-recipe in train.py, tested because every run so far conflated architecture
    # results with an unexamined training-recipe gap (see tasks/todo.md).


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


class WidthGatedMLP(nn.Module):
    """Confidence-gated width: the width-axis analog of a halting head's
    depth-axis question. Same fc1/fc2 shape as the plain dense MLP
    (Linear(d,4d) -> GELU -> Linear(4d,d)), but the SECOND HALF of the 4d
    hidden activations gets scaled by a per-token gate g=sigmoid(Linear
    (d_model,1)(x)) before the second projection -- the first half is
    always fully active (guaranteed baseline capacity, same idea as
    halting always running at least one block), the second half is used
    only as much as the gate says a token needs. last_gate_mean is a
    side-effect attribute set every forward() call (read by
    TinyGPT.width_sparsity_loss, not threaded through Block.forward's
    return signature -- same non-invasive pattern MoEFFN.last_aux_loss
    already uses).
    """

    def __init__(self, d_model: int, use_bitlinear: bool = False):
        super().__init__()
        Linear = BitLinear if use_bitlinear else nn.Linear
        self.d_model = d_model
        self.fc1 = Linear(d_model, 4 * d_model)
        self.fc2 = Linear(4 * d_model, d_model)
        self.gate = nn.Linear(d_model, 1)
        self.last_gate_mean = torch.tensor(0.0)

    def forward(self, x):
        h = F.gelu(self.fc1(x))
        g = torch.sigmoid(self.gate(x))  # (B, T, 1) -- per-token width fraction for the second half
        self.last_gate_mean = g.mean()
        half = self.d_model * 2
        h = torch.cat([h[..., :half], g * h[..., half:]], dim=-1)
        return self.fc2(h)


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
            # selective_decay_layers, if non-empty, specifies exactly which non-attention
            # layers use SelectiveTimeMixing, overriding use_selective_decay's all-or-nothing;
            # empty (default) falls back to that boolean's exact existing behavior.
            if cfg.selective_decay_layers:
                is_selective = layer_idx in cfg.selective_decay_layers
            else:
                is_selective = cfg.use_selective_decay
            if is_selective:
                # Selective (Mamba/S6-style, input-dependent decay) variant --
                # see mamba_lite.py. Same O(1)-state recurrence mechanism as
                # TimeMixing below, just a different (input-dependent) decay.
                self.time_mixing = SelectiveTimeMixing(cfg.d_model, linear_cls=Linear)
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
        elif cfg.use_width_gating:
            self.mlp = WidthGatedMLP(cfg.d_model, cfg.use_bitlinear)
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
        if cfg.tie_layers:
            assert not (cfg.use_rwkv_hybrid and cfg.attention_layers), (
                "tie_layers needs a homogeneous block type -- set attention_layers=() with "
                "use_rwkv_hybrid, or leave use_rwkv_hybrid=False"
            )
            shared_block = Block(cfg, layer_idx=0)
            # Same Python object referenced n_layer times, not n_layer independent instances:
            # nn.Module.parameters() de-duplicates by tensor identity, so num_params() and the
            # optimizer both see this block's weights once, not n_layer times, while
            # hidden_states()'s existing per-index loop still runs it (and tracks a separate
            # RWKV state, where applicable) at every depth step.
            self.blocks = nn.ModuleList([shared_block for _ in range(cfg.n_layer)])
        else:
            self.blocks = nn.ModuleList([Block(cfg, layer_idx=i) for i in range(cfg.n_layer)])
        self.ln_f = nn.LayerNorm(cfg.d_model)
        if not self.use_factored_embedding:
            self.lm_head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
            self.lm_head.weight = self.tok_emb.weight  # tied, standard practice

        self.halt_heads = None
        if cfg.use_halting:
            self.halt_heads = nn.ModuleList([nn.Linear(cfg.d_model, 1) for _ in range(cfg.n_layer)])

        self.extra_heads = None
        if cfg.n_future > 1:
            self.extra_heads = nn.ModuleList(
                [nn.Linear(cfg.d_model, cfg.vocab_size) for _ in range(cfg.n_future - 1)]
            )

        self.proj_head = None
        if cfg.proj_dim > 0:
            self.proj_head = nn.Linear(cfg.d_model, cfg.proj_dim)

        self.apply(self._init_weights)
        if cfg.scaled_residual_init:
            std = 0.02 / math.sqrt(2 * cfg.n_layer)
            for name, p in self.named_parameters():
                # attn_out (attention) and each block's final MLP projection
                # (dense Sequential's index-2 Linear, or WidthGatedMLP/Expert's
                # fc2) are the last linear in their residual branch -- the
                # exact set the GPT-2 paper scales down. MoE experts skipped
                # (architecture question, not this recipe test).
                if name.endswith("attn_out.weight") or name.endswith("fc2.weight") or name.endswith("mlp.2.weight"):
                    nn.init.normal_(p, mean=0.0, std=std)

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

    def width_sparsity_loss(self) -> torch.Tensor:
        """Mean confidence-gated width usage across all WidthGatedMLP blocks
        -- read directly off each block's mlp.last_gate_mean (set as a
        side effect during that block's own forward(), same pattern as
        MoEFFN.last_aux_loss), not threaded through Block.forward's return
        signature. Always safe to call: returns 0.0 if no block is
        width-gated. train.py weights this by WIDTH_SPARSITY_WEIGHT -- the
        only pressure pushing gates below 1.0, since the task loss alone
        has no other reason to.
        """
        gates = [b.mlp.last_gate_mean for b in self.blocks if isinstance(b.mlp, WidthGatedMLP)]
        if not gates:
            return torch.tensor(0.0)
        return torch.stack(gates).mean()

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

    def hidden_states_all_layers(self, idx: torch.Tensor, rwkv_states=None):
        """Same as hidden_states(), but also returns the raw (pre-ln_f) hidden
        state after every block, not just the final one -- needed for the
        halting mechanism (and matches what eval_early_exit.py's untrained
        probe inspected manually; this is the trainable version of the same
        instrumentation).
        """
        B, T = idx.shape
        pos = torch.arange(T, device=idx.device)
        tok_x = self.factored_emb.embed(idx) if self.use_factored_embedding else self.tok_emb(idx)
        x = tok_x + self.pos_emb(pos)
        aux_loss = torch.tensor(0.0, device=idx.device)
        new_states = []
        layer_hiddens = []
        for i, block in enumerate(self.blocks):
            state = rwkv_states[i] if rwkv_states is not None else None
            x, block_aux, new_state = block(x, state)
            aux_loss = aux_loss + block_aux
            new_states.append(new_state)
            layer_hiddens.append(x)
        return layer_hiddens, self.ln_f(x), aux_loss, new_states

    def halting_loss(self, layer_hiddens: list, final_logits: torch.Tensor) -> torch.Tensor:
        """Trains each block's halt_head to predict whether THIS layer's own
        logit-lens prediction (reusing the shared, already-trained ln_f +
        output projection -- only the halt DECISION is being learned here,
        not the projection) already matches the final-depth prediction.
        final_logits should be detached by the caller: the target label is
        an argmax, non-differentiable anyway, but this keeps the halting
        loss from ever influencing the primary task-loss gradient path.
        """
        assert self.halt_heads is not None
        final_pred = final_logits.argmax(dim=-1)  # (B, T)
        project = self.factored_emb.project if self.use_factored_embedding else self.lm_head
        total = torch.tensor(0.0, device=final_logits.device)
        for i, x in enumerate(layer_hiddens[:-1]):  # last layer IS the final prediction, nothing to predict
            probe_logits = project(self.ln_f(x))
            target = (probe_logits.argmax(dim=-1) == final_pred).float()
            halt_logit = self.halt_heads[i](x).squeeze(-1)  # (B, T)
            total = total + F.binary_cross_entropy_with_logits(halt_logit, target)
        return total / max(len(layer_hiddens) - 1, 1)

    def forward_with_halting(self, idx: torch.Tensor):
        """One forward pass returning the standard (logits, extra_logits,
        aux_loss, new_states) plus the halting BCE loss -- avoids a second
        pass through the block stack just to get per-layer hiddens. Only
        meaningful when cfg.use_halting=True (halt_heads is not None).
        """
        layer_hiddens, h, aux_loss, new_states = self.hidden_states_all_layers(idx)
        logits = self.factored_emb.project(h) if self.use_factored_embedding else self.lm_head(h)
        halt_loss = self.halting_loss(layer_hiddens, logits.detach())
        extra_logits = None
        if self.extra_heads is not None:
            extra_logits = [head(h) for head in self.extra_heads]
        return logits, extra_logits, aux_loss, new_states, halt_loss

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
