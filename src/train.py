"""Fast, CPU-first training loop for the ablation harness. One run = one
(dataset, arm, size) point. Every run prints a generated sample partway
through and at the end, so you can eyeball quality, not just read a loss
number -- that eyeballing is as much the point as the scaling curve.

Every checkpoint also reports held-out validation loss, not just train
loss -- train loss alone is misleading for jepa-aux, whose pairs.jsonl is
only ~100 examples and gets revisited dozens of times per run.

Usage:
  python3 src/train.py --dataset rj   --arm base     --size xs
  python3 src/train.py --dataset code --arm mtp      --size s
  python3 src/train.py --dataset code --arm jepa-aux --size m
"""
import argparse
import json
import math
import time
from pathlib import Path

import torch
import torch.nn.functional as F

from codec import N_CODES
from data import (
    AUDIO_OFFSET,
    MARK_RJ,
    MARKERS,
    PIXEL_OFFSET,
    UNIFIED_VOCAB_SIZE,
    get_joint_batch,
    get_lm_batch,
    get_weighted_code_batch,
    load_code_pairs,
    load_joint_modalities,
    load_lm_corpus,
    load_modality_corpus,
    load_scale_up_corpus,
    load_weighted_code_corpus,
    load_weighted_rj_corpus,
)
from model import GPTConfig, TinyGPT
from tokenizer import VOCAB_SIZE, Tokenizer

ROOT = Path(__file__).resolve().parent.parent
RUNS_DIR = ROOT / "runs"

MODALITY_DATASETS = {"pixel", "audio"}  # code-index sequences from codec.py, not BPE text
JOINT_WEIGHTS = {"rj": 0.25, "code": 0.25, "pixel": 0.25, "audio": 0.25}

# n_head chosen to divide d_model; exact counts printed at startup.
SIZES = {
    "xs": dict(d_model=32, n_layer=2, n_head=2),
    "s": dict(d_model=64, n_layer=3, n_head=4),
    "m": dict(d_model=128, n_layer=4, n_head=4),
    "l": dict(d_model=256, n_layer=6, n_head=8),
    # "Grow the layers" -- double 'l's depth (6->12). At the new 8192 vocab,
    # pair with --embedding-rank 64 (TensorRankEmbedding) to control the
    # embedding table's cost, which is proportionally bigger now that vocab
    # grew 8x: 11.6M params plain vs 10.05M with rank=64 factoring.
    "xl": dict(d_model=256, n_layer=12, n_head=8),
    # Scale-up round: text and code domains now have very different
    # realistic data ceilings (text scales easily via Gutenberg, code's
    # zero-download ceiling is ~52M tokens with local site-packages
    # exhausted) -- growing one shared size for both would over-parameterize
    # code again. "xxl" is sized for text specifically (paired with
    # --embedding-rank 96): at the current vocab=32768, 28.08M params
    # targets a Chinchilla-ideal ~561.6M tokens, close to text's projected
    # ~428M-token post-expansion size (~1.3x under, not 10x+ under like
    # forcing code onto this size would be).
    # Correction, computed for real (not assumed) in a later round: "xl"'s
    # 11.6M params is NOT "reasonably matched" to code's real ~43.3M-token
    # ceiling the way this comment used to claim -- Chinchilla-optimal for
    # that real token count (measured directly: 2,528,834 core +
    # 40,735,106 breadth) is ~2.16M params, ~5x smaller. See "chinchilla_min"
    # below, the size that's actually matched.
    "xxl": dict(d_model=384, n_layer=14, n_head=8),
    # Computed (not guessed) Chinchilla-optimal size for code's real,
    # measured token budget (43,263,940 tokens / 20 = 2,163,197 -- this
    # config's 2,259,584 total params, embedding_rank=32, is within 4.5%).
    # Trained and validated: best_val=3.6779, dramatically better than any
    # toy vocab=32768 size tested at mismatched params (see tasks/ducky.md's
    # "minimum viable size" round) -- direct evidence for matching params
    # to real data over just picking a bigger preset. Pair with
    # --rwkv-hybrid --attention-layers 5 --embedding-rank 32
    # --tokenizer-variant balanced --vocab-size 32768 to reproduce.
    "chinchilla_min": dict(d_model=128, n_layer=6, n_head=4),
    # Computed (not guessed) Chinchilla-optimal size for text's real,
    # measured token budget after the Gutenberg expansion + safe chunked
    # tokenization (safe_tokenize_text.py): 305,451,284 tokens / 20 =
    # 15,272,564 -- this config's 15,021,120 total params, embedding_rank=80,
    # is within 2%. Supersedes "xxl"'s ~428M-token projection, which
    # over-estimated the real post-expansion count (28.08M params would
    # have been ~1.8x over-parameterized for what the corpus actually
    # contains). Pair with --rwkv-hybrid --attention-layers 9
    # --embedding-rank 80 --tokenizer-variant balanced --vocab-size 32768.
    "chinchilla_text": dict(d_model=320, n_layer=10, n_head=8),
    # Computed (not guessed) Chinchilla-optimal size for the real
    # scale-up mix's total token budget (tasks/ducky.md Phase AI):
    # literary (rj+gutenberg, 300,326,357) + code_breadth (41,576,969) +
    # code_core (2,587,556) + conversation (9,015) = 344,499,897 tokens
    # / 20 = 17,224,995 -- this config's 17,487,680 total params,
    # embedding_rank=80, is within 2%. Pair with --rwkv-hybrid
    # --attention-layers 11 --embedding-rank 80 --tokenizer-model-path
    # data/tokenizer/spm_ducky_scale_32768.model --scale-up to reproduce.
    "chinchilla_scaleup": dict(d_model=320, n_layer=12, n_head=8),
}

PROMPTS = {
    "rj": "ROMEO:",
    "text": "ROMEO:",  # rj + curated Gutenberg texts -- rj's own opening prompt still works as a sanity check
    "code": "def ",
}


MOE_AUX_WEIGHT = 0.01  # standard small weight for load-balancing losses
HALT_AUX_WEIGHT = 0.1  # weight for the halting-head BCE loss (model.py's halting_loss) -- higher
# than MOE_AUX_WEIGHT since this is the actual training signal for the halt heads, not a
# regularizer on top of an otherwise-independent mechanism
WIDTH_SPARSITY_WEIGHT = 0.05  # weight pushing WidthGatedMLP's gates below 1.0 (model.py's
# width_sparsity_loss) -- the only pressure to use less width on easy tokens, since the task
# loss alone has no other reason to


def get_lr(step: int, warmup_steps: int, max_steps: int, max_lr: float, min_lr: float) -> float:
    """Linear warmup, then cosine decay to min_lr -- the standard nanoGPT/
    minGPT recipe. Opt-in via --lr-schedule cosine (see argparse below);
    every run before this used a flat args.lr for the whole budget, which
    under-serves larger matrices specifically -- the concern motivating
    this addition ahead of re-running the d_model sweep.
    """
    if step < warmup_steps:
        return max_lr * (step + 1) / warmup_steps
    if step > max_steps:
        return min_lr
    decay_ratio = (step - warmup_steps) / max(1, max_steps - warmup_steps)
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return min_lr + coeff * (max_lr - min_lr)


def compute_lm_loss(model, x, targets, pad_id):
    """targets: (B, T, n_future); k=0 is the standard next-token target."""
    use_halting = getattr(model, "halt_heads", None) is not None
    if use_halting:
        logits, extra_logits, aux_loss, _, halt_loss = model.forward_with_halting(x)
    else:
        logits, extra_logits, aux_loss, _ = model(x)
        halt_loss = torch.tensor(0.0)
    width_loss = model.width_sparsity_loss() if hasattr(model, "width_sparsity_loss") else torch.tensor(0.0)
    all_logits = [logits] + (extra_logits or [])
    total, count = 0.0, 0
    for k, lg in enumerate(all_logits):
        tgt = targets[:, :, k]
        valid = tgt != -1
        if not valid.any():
            continue
        loss_k = F.cross_entropy(
            lg.reshape(-1, lg.size(-1)), tgt.reshape(-1), ignore_index=-1
        )
        total = total + loss_k
        count += 1
    return (total / max(count, 1) + MOE_AUX_WEIGHT * aux_loss + HALT_AUX_WEIGHT * halt_loss
            + WIDTH_SPARSITY_WEIGHT * width_loss)


def info_nce(a: torch.Tensor, b: torch.Tensor, temperature: float = 0.1) -> torch.Tensor:
    """Symmetric in-batch contrastive alignment (CLIP-style) between two
    views of the same examples -- no EMA target encoder needed, collapse
    is discouraged by the in-batch negatives instead.
    """
    a = F.normalize(a, dim=-1)
    b = F.normalize(b, dim=-1)
    logits = a @ b.T / temperature
    labels = torch.arange(a.size(0), device=a.device)
    return (F.cross_entropy(logits, labels) + F.cross_entropy(logits.T, labels)) / 2


def run(args):
    is_modality = args.dataset in MODALITY_DATASETS
    is_joint = args.dataset == "joint"
    if is_modality and args.arm == "jepa-aux":
        raise ValueError("jepa-aux needs paired text views (docstring/function) -- not wired for pixel/audio")
    if is_joint and args.arm != "base":
        raise ValueError("joint dataset only wired for the base arm in this pass")

    tok = None if is_modality else (
        Tokenizer(model_path=args.tokenizer_model_path) if args.tokenizer_model_path else
        Tokenizer(vocab_size=args.vocab_size or VOCAB_SIZE, variant=args.tokenizer_variant)
        if (args.vocab_size or args.tokenizer_variant) else Tokenizer()
    )
    vocab_size = UNIFIED_VOCAB_SIZE if is_joint else (N_CODES if is_modality else tok.vocab_size)
    size_cfg = SIZES[args.size]
    n_future = args.n_future if args.arm == "mtp" else 1
    cfg = GPTConfig(
        vocab_size=vocab_size,
        block_size=args.block_size,
        n_future=n_future,
        proj_dim=64 if args.arm == "jepa-aux" else 0,
        moe_experts=args.moe_experts,
        moe_top_k=args.moe_top_k,
        use_bitlinear_experts=args.bitlinear_experts,
        use_rwkv_hybrid=args.rwkv_hybrid,
        attention_layers=tuple(args.attention_layers) if args.attention_layers else (),
        use_bitlinear=args.use_bitlinear,
        embedding_rank=args.embedding_rank,
        tie_layers=args.tie_layers,
        use_halting=args.use_halting,
        use_selective_decay=args.selective_decay,
        selective_decay_layers=tuple(args.selective_decay_layers) if args.selective_decay_layers else (),
        use_width_gating=args.use_width_gating,
        scaled_residual_init=args.nanogpt_recipe,
        **size_cfg,
    )
    device = torch.device(args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu"))
    model = TinyGPT(cfg).to(device)
    n_params = model.num_params()
    eval_bs = args.eval_batch_size if args.eval_batch_size else args.batch_size
    print(f"device: {device}")
    print(f"[{args.dataset}/{args.arm}/{args.size}] {n_params:,} params, block_size={args.block_size}, vocab={vocab_size}")

    # Full-model compilation: bigger speedup than compiling just the WKV
    # scan (0.146s/step vs 0.217s/step, measured -- actually faster than
    # plain dense's uncompiled 0.15s/step), but a much steeper one-time
    # compile cost (~480s vs ~170s). Breakeven is ~4300 extra steps versus
    # scan-only compilation, so this is opt-in for long runs, not the
    # default for quick experiments. compute_model is what the training/
    # eval hot loop calls; model itself (uncompiled) still handles
    # generate()/state_dict()/save -- same underlying parameters either way.
    compute_model = torch.compile(model) if args.compile_full_model else model

    if args.nanogpt_recipe:
        # nanoGPT/uchi (flux/train_v2.py) convention: decay only >=2D weight
        # matrices, never biases/LayerNorm/1D params; betas=(0.9, 0.95) --
        # uchi's own comment: "slightly lower beta2 for stability with small
        # models," the exact regime every run here trains in. Previously
        # this call passed no weight_decay/betas at all, so AdamW silently
        # used its own defaults (weight_decay=0.01 applied to EVERY param
        # including embeddings/LayerNorm/biases, betas=(0.9, 0.999)) --
        # unexamined, not a deliberate choice, and untested against the
        # nanoGPT-standard alternative until now.
        decay, no_decay = [], []
        for p in model.parameters():
            (decay if p.dim() >= 2 else no_decay).append(p)
        opt = torch.optim.AdamW(
            [{"params": decay, "weight_decay": args.weight_decay}, {"params": no_decay, "weight_decay": 0.0}],
            lr=args.lr, betas=(0.9, args.beta2),
        )
    else:
        opt = torch.optim.AdamW(model.parameters(), lr=args.lr)
    min_lr = args.min_lr if args.min_lr is not None else args.lr * 0.1

    moe_suffix = f"_moe{args.moe_experts}" if args.moe_experts > 0 else ""
    moe_suffix += "bl" if args.bitlinear_experts else ""
    moe_suffix += "_rwkv" if args.rwkv_hybrid else ""
    moe_suffix += "_bitlin" if args.use_bitlinear else ""
    moe_suffix += f"_rank{args.embedding_rank}" if args.embedding_rank > 0 else ""
    moe_suffix += "_tied" if args.tie_layers else ""
    moe_suffix += "_halt" if args.use_halting else ""
    moe_suffix += "_selective" if args.selective_decay else ""
    moe_suffix += f"_selectiveL{''.join(map(str, args.selective_decay_layers))}" if args.selective_decay_layers else ""
    moe_suffix += "_widthgate" if args.use_width_gating else ""
    moe_suffix += f"_tok{args.tokenizer_variant}" if args.tokenizer_variant else ""
    moe_suffix += f"_tok{Path(args.tokenizer_model_path).stem}" if args.tokenizer_model_path else ""
    moe_suffix += f"_convw{args.rj_conversation_weight}" if args.rj_conversation_weight is not None else ""
    moe_suffix += f"_codew{args.rj_code_weight}" if args.rj_code_weight is not None else ""
    moe_suffix += "_scaleup" if args.scale_up else ""
    moe_suffix += f"_noo{args.scaleup_noosphere_weight}" if args.scale_up else ""
    moe_suffix += f"_lr{args.lr_schedule}" if args.lr_schedule != "none" else ""
    moe_suffix += "_nanogpt" if args.nanogpt_recipe else ""
    moe_suffix += f"_seed{args.seed}" if args.seed != 0 else ""
    run_name = f"{args.dataset}_{args.arm}_{args.size}{moe_suffix}"
    run_dir = RUNS_DIR / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    metrics_log = []
    samples_log = []

    pad_id = 0 if is_modality else tok.sp.pad_id()

    def lm_batch_loss(ids, bs):
        x, targets = get_lm_batch(ids, bs, args.block_size, n_future)
        x, targets = x.to(device), targets.to(device)
        loss = compute_lm_loss(compute_model, x, targets, pad_id)
        return loss, {"loss": loss.item()}

    def jepa_pair_loss(docs, codes):
        bs = docs.size(0)
        code_targets = torch.full((bs, args.block_size, 1), -1, dtype=torch.long, device=device)
        code_targets[:, :-1, 0] = codes[:, 1:]
        code_targets[codes == pad_id] = -1
        lm_loss = compute_lm_loss(compute_model, codes, code_targets, pad_id)

        doc_targets = torch.full((bs, args.block_size, 1), -1, dtype=torch.long, device=device)
        doc_targets[:, :-1, 0] = docs[:, 1:]
        doc_targets[docs == pad_id] = -1
        doc_lm_loss = compute_lm_loss(compute_model, docs, doc_targets, pad_id)

        doc_emb = model.pooled_embedding(docs, pad_id)
        code_emb = model.pooled_embedding(codes, pad_id)
        align_loss = info_nce(doc_emb, code_emb)

        total = lm_loss + doc_lm_loss + args.align_weight * align_loss
        return total, {
            "code_lm_loss": lm_loss.item(),
            "doc_lm_loss": doc_lm_loss.item(),
            "align_loss": align_loss.item(),
        }

    def jepa_batch(pairs, bs):
        idxs = torch.randint(0, len(pairs), (bs,))
        docs = torch.stack([pairs[i][0] for i in idxs])
        codes = torch.stack([pairs[i][1] for i in idxs])
        return docs, codes

    if args.arm == "jepa-aux":
        train_pairs, val_pairs = load_code_pairs(tok, args.block_size)

        def train_step():
            docs, codes = jepa_batch(train_pairs, args.batch_size)
            return jepa_pair_loss(docs.to(device), codes.to(device))

        @torch.no_grad()
        def eval_step():
            model.eval()
            # val_pairs is tiny (~10) -- use all of it every time, not a sample
            docs = torch.stack([p[0] for p in val_pairs]).to(device)
            codes = torch.stack([p[1] for p in val_pairs]).to(device)
            _, metrics = jepa_pair_loss(docs, codes)
            model.train()
            return {f"val_{k}": v for k, v in metrics.items()}

    elif is_joint:
        joint = load_joint_modalities(tok)
        train_dict = {name: ids for name, (ids, _) in joint.items()}
        val_dict = {name: ids for name, (_, ids) in joint.items()}

        def train_step():
            x, targets = get_joint_batch(train_dict, JOINT_WEIGHTS, args.batch_size, args.block_size)
            x, targets = x.to(device), targets.to(device)
            loss = compute_lm_loss(compute_model, x, targets, pad_id)
            return loss, {"loss": loss.item()}

        @torch.no_grad()
        def eval_step(n_batches=5):
            model.eval()
            per_modality = {name: [] for name in joint}
            for _ in range(n_batches):
                for name in joint:
                    x, targets = get_joint_batch({name: val_dict[name]}, {name: 1.0}, eval_bs, args.block_size)
                    x, targets = x.to(device), targets.to(device)
                    per_modality[name].append(compute_lm_loss(compute_model, x, targets, pad_id).item())
            model.train()
            result = {f"val_loss_{name}": sum(v) / len(v) for name, v in per_modality.items()}
            result["val_loss"] = sum(result.values()) / len(result)
            return result

    else:
        use_weighted_code = args.dataset == "code" and args.code_core_weight is not None
        use_weighted_rj = args.dataset == "rj" and args.rj_conversation_weight is not None
        use_scale_up = args.dataset == "rj" and args.scale_up
        use_weighted_pools = use_weighted_code or use_weighted_rj or use_scale_up
        if is_modality:
            train_ids, val_ids = load_modality_corpus(args.dataset)
        elif use_weighted_code:
            # Stdlib ('core', ~11MB, simple utility-style) vs site-packages
            # ('breadth', ~150MB, dominated by ML/scientific library
            # internals) -- uniform sampling over the concatenated corpus
            # would make core a ~6.6%-by-volume rounding error. Weighted
            # per-example pool choice instead, same pattern as
            # get_joint_batch already uses for pixel/audio. A third
            # 'synthetic' pool (deterministic call-graph compositions,
            # validated to teach real generalization) is added only if
            # data/code/synthetic_relational.txt exists.
            pools = load_weighted_code_corpus(tok)
            if "synthetic" in pools:
                s = args.code_synthetic_weight
                pool_weights = {"core": args.code_core_weight * (1 - s),
                                "breadth": (1 - args.code_core_weight) * (1 - s),
                                "synthetic": s}
            else:
                pool_weights = {"core": args.code_core_weight, "breadth": 1 - args.code_core_weight}
            train_pools = {k: v[0] for k, v in pools.items()}
            val_pools = {k: v[1] for k, v in pools.items()}
        elif use_weighted_rj:
            # Romeo & Juliet vs. the small curated conversation_corpus.txt,
            # optionally plus the stdlib code corpus -- same reasoning as
            # the code core/breadth/synthetic split, see
            # load_weighted_rj_corpus's own docstring. --rj-code-weight is
            # taken off the top (like --code-synthetic-weight), remainder
            # split between rj/conversation per --rj-conversation-weight.
            include_code = args.rj_code_weight is not None
            pools = load_weighted_rj_corpus(tok, include_code=include_code)
            if include_code and "code" in pools:
                c = args.rj_code_weight
                pool_weights = {"rj": (1 - args.rj_conversation_weight) * (1 - c),
                                "conversation": args.rj_conversation_weight * (1 - c),
                                "code": c}
            else:
                pool_weights = {"rj": 1 - args.rj_conversation_weight, "conversation": args.rj_conversation_weight}
            train_pools = {k: v[0] for k, v in pools.items()}
            val_pools = {k: v[1] for k, v in pools.items()}
        elif use_scale_up:
            # Real scale-up mix (tasks/ducky.md Phase AI/AJ): rj / gutenberg
            # (now SEPARATE pools, not one combined "literary" -- Phase AJ
            # fix for gutenberg's general prose diluting rj's own dramatic
            # voice) / conversation / code_core+code_breadth / noosphere.
            # --scaleup-code-weight taken off the top, split internally by
            # --code-core-weight (same composition pattern as
            # --code-synthetic-weight); --scaleup-conversation-weight and
            # --scaleup-noosphere-weight also taken off the top; the
            # remainder is split between rj/gutenberg per --scaleup-rj-
            # weight (rj upweighted within its own share, same stratified-
            # resampling reasoning as the tokenizer's own fix for the
            # identical rj-vs-gutenberg volume disparity). noosphere gets
            # a small fixed share of its own (default 0.05) rather than
            # being split internally like code -- it's a single coherent
            # ~908KB source (workspace/noosphere v1+v2), not two pools.
            pools = load_scale_up_corpus(tok)
            cw = args.scaleup_code_weight
            vw = args.scaleup_conversation_weight
            nw = args.scaleup_noosphere_weight
            text_share = 1 - cw - vw - nw
            rw = args.scaleup_rj_weight
            pool_weights = {
                "code_core": cw * args.code_core_weight,
                "code_breadth": cw * (1 - args.code_core_weight),
                "conversation": vw,
                "rj": text_share * rw,
                "gutenberg": text_share * (1 - rw),
                "noosphere": nw,
            }
            train_pools = {k: v[0] for k, v in pools.items()}
            val_pools = {k: v[1] for k, v in pools.items()}
        else:
            train_ids, val_ids = load_lm_corpus(args.dataset, tok)

        def train_step():
            if use_weighted_pools:
                x, targets = get_weighted_code_batch(train_pools, pool_weights, args.batch_size,
                                                      args.block_size, n_future)
                x, targets = x.to(device), targets.to(device)
                loss = compute_lm_loss(compute_model, x, targets, pad_id)
                return loss, {"loss": loss.item()}
            return lm_batch_loss(train_ids, args.batch_size)

        @torch.no_grad()
        def eval_step(n_batches=5):
            model.eval()
            if use_weighted_pools:
                losses = []
                for _ in range(n_batches):
                    x, targets = get_weighted_code_batch(val_pools, pool_weights, eval_bs,
                                                          args.block_size, n_future)
                    x, targets = x.to(device), targets.to(device)
                    losses.append(compute_lm_loss(compute_model, x, targets, pad_id).item())
            else:
                losses = [lm_batch_loss(val_ids, eval_bs)[1]["loss"] for _ in range(n_batches)]
            model.train()
            return {"val_loss": sum(losses) / len(losses)}

    if is_joint:
        prompt_ids = None  # generated per-modality-marker below instead of one fixed prompt
    elif is_modality:
        prompt_ids = train_ids[:8].unsqueeze(0).to(device)
    else:
        prompt_ids = torch.tensor([tok.encode(PROMPTS[args.dataset])], dtype=torch.long, device=device)

    best_val = float("inf")
    best_step = 0
    patience_counter = 0
    stopped_early = False
    # Plateau state -- separate from patience_counter (early stopping) on purpose:
    # --patience/--min-delta already answers "when to give up," this answers "when
    # to slow down." Independent thresholds so e.g. --plateau-patience 3
    # --patience 10 can decay LR twice before finally stopping. No max_steps
    # horizon needed anywhere in this branch, unlike --lr-schedule cosine --
    # the exact bug that produced two false-negative recipe comparisons this
    # session (rj @ 700 steps, code @ 1200 steps: cosine decayed to its floor
    # before the model had actually converged, because the true horizon was
    # guessed wrong both times). Plateau reacts to the model's own live val-loss
    # trend instead of a schedule fixed in advance.
    plateau_lr = args.lr
    plateau_stall = 0
    plateau_cooldown_left = 0
    t0 = time.time()
    for step in range(1, args.steps + 1):
        current_lr = args.lr
        if args.lr_schedule == "cosine":
            current_lr = get_lr(step - 1, args.warmup_steps, args.steps, args.lr, min_lr)
            for g in opt.param_groups:
                g["lr"] = current_lr
        elif args.lr_schedule == "plateau":
            current_lr = plateau_lr
            for g in opt.param_groups:
                g["lr"] = current_lr

        model.train()
        opt.zero_grad()
        for micro in range(args.grad_accum_steps):
            loss, extra = train_step()
            (loss / args.grad_accum_steps).backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()

        if step % args.log_every == 0 or step == args.steps:
            eval_metrics = eval_step()
            entry = {"step": step, "wall_s": round(time.time() - t0, 2), "lr": current_lr,
                      **extra, **eval_metrics}
            metrics_log.append(entry)
            print(entry)

            val_key = "val_loss" if "val_loss" in eval_metrics else "val_code_lm_loss"
            if eval_metrics[val_key] < best_val - args.min_delta:
                best_val = eval_metrics[val_key]
                best_step = step
                patience_counter = 0
                plateau_stall = 0
                torch.save({k: v.cpu() for k, v in model.state_dict().items()}, run_dir / "model_best.pt")
            else:
                patience_counter += 1
                if args.lr_schedule == "plateau":
                    if plateau_cooldown_left > 0:
                        plateau_cooldown_left -= 1
                    else:
                        plateau_stall += 1
                        if plateau_stall >= args.plateau_patience:
                            new_lr = max(plateau_lr * args.plateau_factor, min_lr)
                            if new_lr < plateau_lr:
                                print(f"plateau: reducing lr {plateau_lr:.2e} -> {new_lr:.2e} "
                                      f"at step {step} (no improvement for {args.plateau_patience} checkpoints)")
                                plateau_lr = new_lr
                                # A decay's whole point is to give the model a fresh
                                # shot at a lower LR -- previously patience_counter
                                # kept counting through it unchanged, so whenever
                                # --patience was only slightly bigger than
                                # --plateau-patience (as in this project's own
                                # "production" defaults, 6 vs 5), early stopping
                                # fired ~1 checkpoint after the decay, before the
                                # new LR could possibly show any improvement.
                                # Found via a real production run: decay at step
                                # 11000, early-stop at step 11250, final loss
                                # measurably worse than a longer un-decayed run.
                                patience_counter = 0
                            plateau_stall = 0
                            plateau_cooldown_left = args.plateau_cooldown
                # Every extended sweep this session found its real ceiling
                # by running long past it and reading off the best
                # checkpoint after the fact -- e.g. code/base/m at step
                # 1250 needed a 2000-step run to discover. Patience=0
                # disables this (matches every run so far, all of which
                # used a fixed --steps budget with no early stop).
                if args.patience > 0 and patience_counter >= args.patience:
                    print(f"early stopping at step {step}: no improvement for "
                          f"{args.patience} checkpoints (best was step {best_step}: {best_val:.4f})")
                    stopped_early = True
                    break

        if step % args.sample_every == 0 or step == args.steps:
            if is_joint:
                # The interesting question here isn't fluency, it's whether
                # the model respects modality boundaries: seeded with only
                # a marker token, does it keep emitting ids in that
                # modality's range, or drift into another one's?
                names_by_marker = {v: k for k, v in MARKERS.items()}

                def describe(ids):
                    parts = []
                    for tid in ids:
                        if tid in names_by_marker:
                            parts.append(f"<{names_by_marker[tid].upper()}>")
                        elif tid < PIXEL_OFFSET:
                            parts.append(tok.decode([tid]))
                        elif tid < AUDIO_OFFSET:
                            parts.append(f"[P{tid - PIXEL_OFFSET}]")
                        else:
                            parts.append(f"[A{tid - AUDIO_OFFSET}]")
                    return " ".join(parts)

                def in_lane_frac(ids, name):
                    gen = ids[1:]  # drop the seed marker itself
                    if name in ("rj", "code"):
                        hits = sum(t < PIXEL_OFFSET for t in gen)
                    elif name == "pixel":
                        hits = sum(PIXEL_OFFSET <= t < AUDIO_OFFSET for t in gen)
                    else:
                        hits = sum(AUDIO_OFFSET <= t < MARK_RJ for t in gen)
                    return hits / len(gen)

                text_parts = []
                for name, marker in MARKERS.items():
                    seed = torch.tensor([[marker]], dtype=torch.long, device=device)
                    out = model.generate(seed, max_new_tokens=40)[0].tolist()
                    frac = in_lane_frac(out, name)
                    text_parts.append(f"{name} (in-lane {frac:.0%}): {describe(out)}")
                text = " | ".join(text_parts)
            elif is_modality:
                out = model.generate(prompt_ids.clone(), max_new_tokens=60)
                # No text rendering for code-index sequences in this chat --
                # print the raw codes (not directly visualizable/audible
                # here; codec.py's decoder can reconstruct an image/wav from
                # them if that's ever needed).
                text = str(out[0].tolist())
            else:
                out = model.generate(prompt_ids.clone(), max_new_tokens=60)
                text = tok.decode(out[0].tolist())
            samples_log.append({"step": step, "sample": text})
            print(f"  sample @ {step}: {text!r}")

    (run_dir / "metrics.json").write_text(json.dumps(metrics_log, indent=2))
    (run_dir / "samples.json").write_text(json.dumps(samples_log, indent=2))
    torch.save({k: v.cpu() for k, v in model.state_dict().items()}, run_dir / "model_final.pt")
    (run_dir / "config.json").write_text(
        json.dumps({**vars(args), "n_params": n_params, "best_step": best_step, "best_val": best_val,
                     "vocab_size": vocab_size}, indent=2)
    )
    total_wall = time.time() - t0
    print(f"done in {total_wall:.1f}s -> {run_dir} (best@{best_step}: {best_val:.4f})")

    best_entry = next((e for e in metrics_log if e["step"] == best_step), metrics_log[-1])
    return {
        "dataset": args.dataset,
        "arm": args.arm,
        "size": args.size,
        "n_params": n_params,
        "wall_s": round(total_wall, 2),
        "best_step": best_step,
        **best_entry,
    }


def build_parser() -> argparse.ArgumentParser:
    """Factored out of __main__ so run_hpo_sweep.py can build the exact same
    argparse.Namespace train.run() expects (many attributes -- reconstructing
    them by hand would drift out of sync with this file) instead of
    duplicating the CLI surface or shelling out per trial.
    """
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", choices=["rj", "text", "code", "pixel", "audio", "joint"], required=True)
    p.add_argument("--arm", choices=["base", "mtp", "jepa-aux"], required=True)
    p.add_argument("--size", choices=list(SIZES), default="xs")
    p.add_argument("--block-size", type=int, default=128)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--grad-accum-steps", type=int, default=1, help="accumulate gradients over this many "
                    "micro-batches (each of --batch-size) before opt.step() -- lets a smaller --batch-size "
                    "fit in less GPU memory while keeping the same effective batch size (batch_size * "
                    "grad_accum_steps). Added to run alongside a concurrent GPU job without OOM; default 1 "
                    "is byte-for-byte identical to every existing run's behavior.")
    p.add_argument("--eval-batch-size", type=int, default=None, help="batch size used for validation only; "
                    "defaults to --batch-size (old behavior, unchanged) if not set. Previously eval_step "
                    "always reused --batch-size, so shrinking it for --grad-accum-steps (memory reasons) "
                    "silently made validation noisier too (fewer samples per checkpoint) -- this decouples "
                    "the two, so a small training micro-batch doesn't have to mean a noisy val-loss estimate.")
    p.add_argument("--steps", type=int, default=300)
    p.add_argument("--patience", type=int, default=0, help="stop after this many checkpoints with no val "
                    "improvement (0 = disabled, run the full --steps budget -- matches every run this session "
                    "so far, all of which used a fixed budget with no early stop)")
    p.add_argument("--min-delta", type=float, default=0.0, help="minimum val-loss improvement to reset patience")
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--n-future", type=int, default=2, help="mtp arm only")
    p.add_argument("--align-weight", type=float, default=0.5, help="jepa-aux arm only")
    p.add_argument("--log-every", type=int, default=25)
    p.add_argument("--sample-every", type=int, default=100)
    p.add_argument("--moe-experts", type=int, default=0, help="0 = dense MLP; >0 = MoE FFN with this many routed experts")
    p.add_argument("--moe-top-k", type=int, default=1)
    p.add_argument("--bitlinear-experts", action="store_true", help="quantize MoE experts (ported from uchi's BitLinear)")
    p.add_argument("--rwkv-hybrid", action="store_true", help="RWKV time-mixing blocks + periodic attention, instead of pure attention")
    p.add_argument("--attention-layers", type=int, nargs="*", default=[], help="0-indexed layers using attention when --rwkv-hybrid; rest use RWKV")
    p.add_argument("--use-bitlinear", action="store_true", help="BitLinear throughout Ducky's own blocks (attention/RWKV + dense MLP)")
    p.add_argument("--embedding-rank", type=int, default=0, help="0 = plain tied embedding; >0 = TensorRankEmbedding at this rank")
    p.add_argument("--tie-layers", action="store_true", help="Universal-Transformer/ALBERT-style: reuse one "
                    "physical block's weights at every depth step instead of n_layer independent blocks -- "
                    "cuts the block stack's param count, requires a homogeneous block type (see model.py)")
    p.add_argument("--vocab-size", type=int, default=None, help="explicit tokenizer vocab size (e.g. 1024 to "
                    "match older toy-scale checkpoints) -- default None uses tokenizer.py's module default "
                    "(currently 32768), unchanged existing behavior")
    p.add_argument("--tokenizer-model-path", type=str, default=None, help="load an exact .model file via "
                    "Tokenizer(model_path=...), bypassing the vocab_size/variant naming convention entirely -- "
                    "for a dedicated single-corpus tokenizer (e.g. one trained only on romeo_and_juliet.txt) "
                    "that isn't part of the shared multi-domain vocab family. Takes precedence over "
                    "--vocab-size/--tokenizer-variant if both are given.")
    p.add_argument("--selective-decay", action="store_true", help="mamba_lite.py's SelectiveTimeMixing "
                    "(input-dependent decay) instead of rwkv_model.py's TimeMixing for non-attention "
                    "blocks when --rwkv-hybrid is set")
    p.add_argument("--selective-decay-layers", type=int, nargs="*", default=[], help="0-indexed "
                    "non-attention layers that use SelectiveTimeMixing specifically -- overrides "
                    "--selective-decay's all-or-nothing when given (e.g. attention-adjacent layers only)")
    p.add_argument("--use-width-gating", action="store_true", help="WidthGatedMLP instead of the plain "
                    "dense MLP -- confidence-gated width, the width-axis analog of --use-halting's "
                    "depth-axis question (model.py's WidthGatedMLP/width_sparsity_loss)")
    p.add_argument("--use-halting", action="store_true", help="per-block halting head, trained via an "
                    "auxiliary BCE loss (model.py's halting_loss) to predict whether this layer's own "
                    "prediction already matches the final-depth one -- the trained counterpart to "
                    "eval_early_exit.py's untrained logit-lens probe")
    p.add_argument("--tokenizer-variant", type=str, default="", help="tokenizer.py's Tokenizer(variant=...) -- "
                    "e.g. 'balanced' to opt into data/tokenizer/spm_{vocab_size}_balanced.model (see "
                    "build_production_tokenizer.py) instead of the default naive-concat tokenizer. Plumbing only: "
                    "does not itself retrain anything, just enables a future run to request the new tokenizer.")
    p.add_argument("--code-core-weight", type=float, default=0.5,
                    help="--dataset code only: sampling weight for stdlib ('core') vs site-packages "
                    "('breadth') per training example, default 0.5/0.5 despite core being ~14x smaller "
                    "by raw size -- without this, core's simple-utility style becomes a rounding error "
                    "during training. Set to None (via code, not CLI) to disable and use plain uniform "
                    "sampling over the concatenated corpus instead.")
    p.add_argument("--code-synthetic-weight", type=float, default=0.15,
                    help="--dataset code only, applied on top of --code-core-weight when "
                    "data/code/synthetic_relational.txt exists: fraction of training examples drawn "
                    "from synthetic call-graph-composed statements, remainder split between core/breadth "
                    "per --code-core-weight.")
    p.add_argument("--rj-conversation-weight", type=float, default=None,
                    help="--dataset rj only: sampling weight for the curated conversation_corpus.txt pool "
                    "per training example (rj gets 1 - this, or 1 - this - rj-code-weight's remainder if "
                    "--rj-code-weight is also set). Default None disables weighted pooling entirely (plain "
                    "rj-only training, unchanged existing behavior) -- conversation_corpus.txt is opt-in, "
                    "not a silent default change.")
    p.add_argument("--rj-code-weight", type=float, default=None,
                    help="--dataset rj only, requires --rj-conversation-weight also set: sampling weight "
                    "for the stdlib code corpus (corpus_core.txt) taken off the top, remainder split "
                    "between rj/conversation per --rj-conversation-weight -- same composition pattern as "
                    "--code-synthetic-weight over --code-core-weight. Default None disables the code pool "
                    "entirely (two-pool rj/conversation training, unchanged existing behavior).")
    p.add_argument("--scale-up", action="store_true",
                    help="--dataset rj only: real scale-up mix (tasks/ducky.md Phase AI/AJ/AK) -- rj / "
                    "gutenberg_corpus.txt (separate pools, not one combined 'literary' -- Phase AJ fix "
                    "for gutenberg's general prose diluting rj's own dramatic voice) / conversation / "
                    "code_core+code_breadth / noosphere (Phase AK: real code pulled from the sibling "
                    "workspace/noosphere project). Requires all six pools already safely tokenized ahead "
                    "of time under whatever --tokenizer-model-path is given -- this flag does not itself "
                    "do any large, unsafe tokenization.")
    p.add_argument("--scaleup-code-weight", type=float, default=0.4,
                    help="--scale-up only: combined sampling weight for code_core+code_breadth, split "
                    "internally by --code-core-weight (same composition as --code-synthetic-weight).")
    p.add_argument("--scaleup-conversation-weight", type=float, default=0.2,
                    help="--scale-up only: sampling weight for the curated conversation pool, taken off "
                    "the remainder after --scaleup-code-weight. The rest (1 - scaleup_code_weight - "
                    "scaleup_conversation_weight) is split between rj/gutenberg per --scaleup-rj-weight.")
    p.add_argument("--scaleup-rj-weight", type=float, default=0.3,
                    help="--scale-up only: fraction of the rj+gutenberg text share given to rj specifically "
                    "(gutenberg gets the rest) -- rj is only ~150KB against gutenberg's ~1.3GB (~8700:1), "
                    "so without real upweighting here rj's own dramatic voice gets diluted into gutenberg's "
                    "general prose (tasks/ducky.md Phase AI's observed register-drift finding). Same "
                    "stratified-resampling discipline as the tokenizer's own rj/conversation upweighting.")
    p.add_argument("--scaleup-noosphere-weight", type=float, default=0.05,
                    help="--scale-up only: fixed top-level sampling weight for the noosphere code pool "
                    "(corpus_noosphere.txt -- real PyTorch/signal-processing source pulled from the sibling "
                    "workspace/noosphere project, ~908KB/23.4K lines). Taken off the top like "
                    "--scaleup-conversation-weight, not split internally like --scaleup-code-weight, since "
                    "it's one coherent source rather than two pools. Small default share since it's tiny "
                    "against code_core+code_breadth -- kept off zero so it gets real training exposure "
                    "rather than rounding away, same reasoning as every other small-pool upweighting this "
                    "project has needed (rj vs. gutenberg, rj+conversation in the tokenizer itself).")
    p.add_argument("--lr-schedule", choices=["none", "cosine", "plateau"], default="none", help="'none' "
                    "(default) -- flat args.lr for the whole run, unchanged behavior for every existing "
                    "checkpoint's reproducibility. 'cosine' -- linear warmup then cosine decay to --min-lr "
                    "over a FIXED --steps horizon (the standard nanoGPT/minGPT recipe) -- twice this session "
                    "produced a false negative when the guessed horizon was shorter than the model's actual "
                    "convergence point (decayed to near-zero LR while still improving under flat LR). "
                    "'plateau' -- reacts to the model's own live val-loss trend instead (ReduceLROnPlateau-"
                    "style: reuses the existing best_val/patience-counter tracking below), no horizon "
                    "guess required -- see --plateau-patience/--plateau-factor/--plateau-cooldown.")
    p.add_argument("--warmup-steps", type=int, default=100, help="only used when --lr-schedule cosine")
    p.add_argument("--min-lr", type=float, default=None, help="used when --lr-schedule cosine or plateau "
                    "as a floor; default None computes args.lr * 0.1, the common nanoGPT ratio")
    p.add_argument("--plateau-patience", type=int, default=5, help="--lr-schedule plateau only: checkpoints "
                    "with no val-loss improvement (independent of --patience's early-stopping counter) "
                    "before multiplying LR by --plateau-factor")
    p.add_argument("--plateau-factor", type=float, default=0.5, help="--lr-schedule plateau only: LR "
                    "multiplier applied on each plateau decay, floored at --min-lr")
    p.add_argument("--plateau-cooldown", type=int, default=2, help="--lr-schedule plateau only: checkpoints "
                    "to wait after a decay before plateau-patience starts counting again, so one decay "
                    "doesn't immediately trigger another before the new LR has had a chance to help")
    p.add_argument("--weight-decay", type=float, default=0.1, help="--nanogpt-recipe only: applied to "
                    ">=2D params only (0.0 on biases/LayerNorm regardless of this value) -- was hardcoded "
                    "0.1 (nanoGPT/uchi's own value), now exposed for run_hpo_sweep.py to search")
    p.add_argument("--beta2", type=float, default=0.95, help="--nanogpt-recipe only: AdamW's second beta "
                    "(beta1 stays fixed at 0.9) -- was hardcoded 0.95, now exposed for run_hpo_sweep.py "
                    "to search")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", type=str, default=None, help="'cuda' or 'cpu'; default auto-detects cuda if "
                    "available. Previously this codebase had no device placement at all -- every tensor "
                    "implicitly lived on CPU regardless of free GPU (measured 30-40x steady-state speedup "
                    "on cuda once past torch.compile's one-time cost, see run_scaling_sweep.py)")
    p.add_argument("--num-threads", type=int, default=8, help="torch.set_num_threads() -- measured optimal on this "
                    "machine (20 logical cores); the default 10 is close but 8 was faster, and 16-20 was 4-6x slower "
                    "from thread-sync overhead swamping the benefit on our small tensor ops")
    p.add_argument("--compile-full-model", action="store_true", help="compile the whole forward pass, not just the "
                    "WKV scan -- faster steady-state (0.146s/step measured vs 0.217s/step scan-only) but ~480s "
                    "one-time compile cost vs ~170s; only worth it for long runs (breakeven ~4300 extra steps)")
    p.add_argument("--nanogpt-recipe", action="store_true", help="bundle of nanoGPT/uchi training-recipe "
                    "fixes never tested against this project's default AdamW(lr=lr) call: betas=(0.9, 0.95) "
                    "instead of torch's (0.9, 0.999), weight_decay=0.1 on >=2D params only (0.0 on biases/"
                    "LayerNorm, instead of AdamW's default 0.01 applied uniformly to everything including "
                    "embeddings), and GPT-2-style scaled residual-projection init (model.py's "
                    "scaled_residual_init). Independent of --lr-schedule -- pass both to match the full "
                    "nanoGPT/uchi recipe. Opt-in, does not change any existing run's reproducibility.")
    return p


if __name__ == "__main__":
    args = build_parser().parse_args()
    torch.manual_seed(args.seed)
    torch.set_num_threads(args.num_threads)
    run(args)
