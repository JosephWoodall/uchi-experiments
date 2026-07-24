"""Instruction/task-format fine-tuning: the untested lever this project's
own history points at. Every checkpoint trained so far (including the
real production DEFAULT_RUN) used plain next-token CE over undifferentiated
corpora -- next, gutenberg, code, conversation all mixed by domain-weighted
sampling, never an objective that explicitly teaches "given a task-shaped
prompt, produce a task-shaped completion." bench_ducky.py's own 0/10 (every
architecture/scale/recipe tried) and the new agent harness's 0/5 (Ducky
never even recognized the Thought/Action format) are both consistent with
that same missing lever, not just coincidentally similar failures.

Fine-tunes (not from-scratch trains) an EXISTING checkpoint on real
docstring->function-body pairs (data/code/pairs.jsonl, 63,607 raw pairs,
already used for jepa-aux's embedding-alignment loss but never for a direct
completion objective) with a prompt-masked cross-entropy loss: only
completion-token positions contribute to the loss, matching standard SFT
convention and exactly bench_ducky.py's own prompt/completion split
(signature+docstring as prompt, body as target) -- verified against 2000
real pairs, 95.5% parse cleanly via ast.FunctionDef + ast.get_docstring
(see this file's own build_sft_examples()).

Small-scale-first, per this project's own no_scaleup_without_proof rule:
fine-tunes the existing 2.26M-param Chinchilla-min code checkpoint
(runs/code_base_chinchilla_min_rwkv_rank32_tokbalanced_lrplateau_nanogpt_
seed57, best_val=3.6423, "coherent but not capable" per tasks/ducky.md
Phase T) first, not the 17.49M-param production DEFAULT_RUN -- proves or
disproves the hypothesis cheaply before spending more compute on it.
"""
import argparse
import ast
import json
import time
from pathlib import Path

import torch
import torch.nn.functional as F

from model import GPTConfig, TinyGPT
from tokenizer import Tokenizer
from train import SIZES

ROOT = Path(__file__).resolve().parent.parent
PAIRS_PATH = ROOT / "data" / "code" / "pairs.jsonl"


def load_model_and_config(run_name: str):
    """Identical loading convention to eval_chinchilla_min.py's own
    load_model() -- reused by name, not duplicated as new logic, so the
    before/after comparison is loading through the exact same path both
    times."""
    cfg_dict = json.loads((ROOT / "runs" / run_name / "config.json").read_text())
    size_cfg = SIZES.get(cfg_dict["size"])
    if size_cfg is None:
        size_cfg = dict(d_model=128, n_layer=6, n_head=4)
    cfg = GPTConfig(
        vocab_size=cfg_dict["vocab_size"],
        block_size=cfg_dict["block_size"],
        use_rwkv_hybrid=cfg_dict["rwkv_hybrid"],
        attention_layers=tuple(cfg_dict["attention_layers"]),
        embedding_rank=cfg_dict["embedding_rank"],
        **size_cfg,
    )
    model = TinyGPT(cfg)
    model.load_state_dict(
        torch.load(ROOT / "runs" / run_name / "model_best.pt", map_location="cpu", weights_only=True)
    )
    return model, cfg_dict


def build_sft_examples(max_pairs: int | None = None) -> list[tuple[str, str]]:
    """(prompt, completion) text pairs -- prompt is everything through the
    function signature + docstring, completion is the body. Uses
    ast.FunctionDef + the docstring Expr statement's own end_lineno to find
    the exact split point, not string search (handles both quote styles,
    multi-line docstrings, correctly) -- same ast-based-parsing discipline
    already established in grounding.py/action_parser.py. Silently skips
    pairs that don't parse as a single function with a real string-literal
    docstring first statement (89/2000 = 4.5% in a real sample: multi-
    statement decorators, missing docstrings, etc.) -- an honest data-
    quality filter, not a bug to chase.
    """
    examples = []
    with PAIRS_PATH.open() as f:
        for i, line in enumerate(f):
            if max_pairs is not None and i >= max_pairs:
                break
            code = json.loads(line)["code"]
            try:
                tree = ast.parse(code)
            except SyntaxError:
                continue
            if not tree.body or not isinstance(tree.body[0], ast.FunctionDef):
                continue
            fn = tree.body[0]
            if not fn.body:
                continue
            first_stmt = fn.body[0]
            if not (
                isinstance(first_stmt, ast.Expr)
                and isinstance(first_stmt.value, ast.Constant)
                and isinstance(first_stmt.value.value, str)
            ):
                continue
            lines = code.splitlines(keepends=True)
            doc_end_line = first_stmt.end_lineno
            prompt = "".join(lines[:doc_end_line])
            completion = "".join(lines[doc_end_line:])
            if completion.strip():
                examples.append((prompt, completion))
    return examples


def encode_sft_batch(examples: list[tuple[str, str]], tok: Tokenizer, block_size: int):
    """Returns (x, targets, mask) each (B, block_size-1) -- mask[i]==1 iff
    targets[i] is a completion token, matching standard SFT prompt-masking:
    predicting the FIRST completion token (given the full prompt as
    context) IS supervised; predicting any prompt token from earlier
    prompt context is not. Examples whose prompt alone already exceeds
    block_size are skipped (nothing sensible to supervise); longer
    completions are truncated, shorter sequences padded with pad_id
    (mask=0 over padding, same as it is over the prompt).
    """
    pad_id = tok.sp.pad_id()
    xs, targets_list, masks = [], [], []
    for prompt, completion in examples:
        prompt_ids = tok.encode(prompt)
        completion_ids = tok.encode(completion)
        if len(prompt_ids) >= block_size:
            continue
        ids = (prompt_ids + completion_ids)[:block_size]
        prompt_len = min(len(prompt_ids), len(ids))

        ids = ids + [pad_id] * (block_size - len(ids))
        x = ids[:-1]
        targets = ids[1:]
        # target[i] is a completion token iff i >= prompt_len - 1 (the
        # target at i=prompt_len-1 is completion_ids[0], predicted from
        # the full prompt as context) AND it's not padding.
        mask = [
            1 if (i >= prompt_len - 1 and targets[i] != pad_id) else 0
            for i in range(len(targets))
        ]
        xs.append(x)
        targets_list.append(targets)
        masks.append(mask)

    return (
        torch.tensor(xs, dtype=torch.long),
        torch.tensor(targets_list, dtype=torch.long),
        torch.tensor(masks, dtype=torch.float),
    )


def masked_ce_loss(logits: torch.Tensor, targets: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """logits: (B, T, V), targets/mask: (B, T). Only mask==1 positions
    contribute -- verified in verify_sft_masking() below against an
    independent same-answer computation (slice to the completion span and
    compute plain CE) before this is trusted in the real training loop.
    """
    per_token = F.cross_entropy(
        logits.reshape(-1, logits.size(-1)), targets.reshape(-1), reduction="none"
    ).reshape(targets.shape)
    denom = mask.sum().clamp(min=1.0)
    return (per_token * mask).sum() / denom


def verify_sft_masking() -> None:
    """Sanity check, same discipline as bench_ducky.py's own harness
    self-verification: masked_ce_loss on a real (prompt, completion) pair,
    computed two independent ways, must agree numerically. If they don't,
    the masking logic is wrong and nothing downstream should be trusted.
    """
    torch.manual_seed(0)
    vocab_size, block_size = 50, 16
    logits = torch.randn(1, block_size - 1, vocab_size, requires_grad=True)

    prompt_len = 6
    completion_len = 5
    total_len = prompt_len + completion_len
    ids = torch.randint(0, vocab_size, (total_len,)).tolist()
    pad_id = 49
    ids = ids[:block_size]
    ids = ids + [pad_id] * (block_size - len(ids))
    targets = torch.tensor(ids[1:]).unsqueeze(0)
    mask = torch.tensor(
        [1.0 if (i >= prompt_len - 1 and targets[0, i].item() != pad_id) else 0.0 for i in range(block_size - 1)]
    ).unsqueeze(0)

    loss_via_mask = masked_ce_loss(logits, targets, mask)

    # Independent computation: slice directly to the known completion span
    # and compute plain, unmasked CE over exactly that slice.
    start = prompt_len - 1
    end = total_len - 1
    logits_slice = logits[:, start:end, :]
    targets_slice = targets[:, start:end]
    loss_via_slice = F.cross_entropy(
        logits_slice.reshape(-1, vocab_size), targets_slice.reshape(-1)
    )

    diff = (loss_via_mask - loss_via_slice).abs().item()
    assert diff < 1e-5, f"masked_ce_loss disagrees with the independent slice computation: diff={diff}"

    # Also verify the mask actually EXCLUDES the prompt region: corrupting
    # prompt-position logits must not change the masked loss at all.
    logits2 = logits.clone().detach().requires_grad_(True)
    with torch.no_grad():
        logits2[:, : prompt_len - 1, :] += 1000.0  # would drastically change an unmasked loss
    loss_corrupted_prompt = masked_ce_loss(logits2, targets, mask)
    diff2 = (loss_via_mask - loss_corrupted_prompt).abs().item()
    assert diff2 < 1e-5, (
        f"masked_ce_loss changed after corrupting ONLY prompt-position logits "
        f"(diff={diff2}) -- the mask is not correctly excluding the prompt."
    )

    print(
        f"[verify_sft_masking] PASS: masked loss matches independent slice computation "
        f"(diff={diff:.2e}), and is invariant to prompt-logit corruption (diff={diff2:.2e})"
    )


def run_sft(
    run_name: str,
    output_run_name: str,
    steps: int = 500,
    batch_size: int = 16,
    lr: float = 1e-4,
    max_pairs: int | None = None,
    log_every: int = 50,
    seed: int = 57,
):
    torch.manual_seed(seed)
    model, cfg_dict = load_model_and_config(run_name)
    tok = Tokenizer(vocab_size=cfg_dict["vocab_size"], variant=cfg_dict.get("tokenizer_variant", ""))
    block_size = cfg_dict["block_size"]

    print("Building SFT examples from data/code/pairs.jsonl...")
    examples = build_sft_examples(max_pairs=max_pairs)
    n_val = max(1, int(len(examples) * 0.1))
    train_examples, val_examples = examples[:-n_val], examples[-n_val:]
    print(f"  {len(train_examples)} train / {len(val_examples)} val SFT examples")

    opt = torch.optim.AdamW(model.parameters(), lr=lr, betas=(0.9, 0.95), weight_decay=0.1)

    model.train()
    rng = torch.Generator().manual_seed(seed)
    for step in range(1, steps + 1):
        idx = torch.randint(0, len(train_examples), (batch_size,), generator=rng)
        batch = [train_examples[i] for i in idx.tolist()]
        x, targets, mask = encode_sft_batch(batch, tok, block_size)
        if x.numel() == 0:
            continue

        logits, _, _, _ = model(x)
        loss = masked_ce_loss(logits, targets, mask)

        opt.zero_grad()
        loss.backward()
        opt.step()

        if step % log_every == 0 or step == 1:
            model.eval()
            with torch.no_grad():
                val_batch = val_examples[: min(batch_size, len(val_examples))]
                vx, vtargets, vmask = encode_sft_batch(val_batch, tok, block_size)
                vlogits, _, _, _ = model(vx)
                vloss = masked_ce_loss(vlogits, vtargets, vmask)
            print(f"  step {step}/{steps} train_loss={loss.item():.4f} val_loss={vloss.item():.4f}")
            model.train()

    out_dir = ROOT / "runs" / output_run_name
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), out_dir / "model_best.pt")
    (out_dir / "config.json").write_text(json.dumps({**cfg_dict, "sft_base_run": run_name, "sft_steps": steps}, indent=2))
    print(f"Saved SFT checkpoint to {out_dir}")
    return out_dir


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run-name", default="code_base_chinchilla_min_rwkv_rank32_tokbalanced_lrplateau_nanogpt_seed57")
    p.add_argument("--output-run-name", default="code_base_chinchilla_min_sft")
    p.add_argument("--steps", type=int, default=500)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--max-pairs", type=int, default=None)
    p.add_argument("--log-every", type=int, default=50)
    p.add_argument("--seed", type=int, default=57)
    p.add_argument("--skip-verify", action="store_true")
    return p


def main():
    args = build_parser().parse_args()
    if not args.skip_verify:
        verify_sft_masking()
    t0 = time.time()
    run_sft(
        run_name=args.run_name,
        output_run_name=args.output_run_name,
        steps=args.steps,
        batch_size=args.batch_size,
        lr=args.lr,
        max_pairs=args.max_pairs,
        log_every=args.log_every,
        seed=args.seed,
    )
    print(f"Total time: {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
