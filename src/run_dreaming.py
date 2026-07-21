"""Self-distillation "dreaming" (Furlanello et al. 2018, "Born-Again
Neural Networks," arXiv:1805.04770; softened-target distillation, Hinton
et al. 2015, arXiv:1503.02531): a frozen teacher (the Chinchilla-matched
code checkpoint from tasks/ducky.md's "minimum viable size" round) free-
runs ("dreams") continuations from real seed contexts, and a student
copy is trained to match the teacher's own SOFTENED distribution on those
dreamed continuations -- consolidating what the model already encodes,
not adding new external facts the way the failed flywheel round would
have. No hard pseudo-labels, no verification gate needed: the target is
the teacher's own (smoothed) belief, not a claim about ground truth.

Deliberately small: a first test of a new idea, not a committed
investment -- a few hundred distillation steps, ~30 dream sequences.
"""
import copy
import json
import time
from pathlib import Path

import torch
import torch.nn.functional as F

from bench_ducky import TASKS, run_task
from data import get_lm_batch
from eval_chinchilla_min import greedy_ask, load_model
from inference import _block_repeat_ngrams
from tokenizer import Tokenizer
from train import compute_lm_loss

ROOT = Path(__file__).resolve().parent.parent
TEACHER_RUN = "code_base_chinchilla_min_rwkv_rank32_tokbalanced"
STUDENT_RUN_NAME = "code_base_chinchilla_min_dream_student"

N_DREAMS = 30
PREFIX_LEN = 40
DREAM_LEN = 40
DISTILL_TEMPERATURE = 2.0  # Hinton et al. 2015
DISTILL_STEPS = 400
BATCH_SIZE = 8  # dreamed sequences are grouped into small batches for the distill step


@torch.no_grad()
def dream(teacher, tok, seed_ids: torch.Tensor, dream_len: int = DREAM_LEN, temperature: float = 0.8) -> list:
    """One free-running continuation from a real seed, sampled (not
    greedy) so repeated dreams from similar seeds don't collapse to
    identical continuations -- the point is exploring around what the
    teacher already believes, not reciting one deterministic answer.
    """
    ids = seed_ids.tolist()
    block_size = teacher.cfg.block_size
    for _ in range(dream_len):
        ctx = torch.tensor([ids[-block_size:]], dtype=torch.long)
        logits, _, _, _ = teacher(ctx)
        step_logits = _block_repeat_ngrams(logits[0, -1, :], ids, 4)
        probs = torch.softmax(step_logits / temperature, dim=-1)
        next_id = torch.multinomial(probs, num_samples=1).item()
        ids.append(next_id)
    return ids


def distill_loss(student_logits: torch.Tensor, teacher_logits: torch.Tensor, prefix_len: int,
                  temperature: float = DISTILL_TEMPERATURE) -> torch.Tensor:
    """KL(student || teacher) on the DREAMED portion only (positions >=
    prefix_len -- the real seed tokens already have real next-token labels
    from training, nothing to distill there), both softened by
    `temperature` (Hinton et al. 2015's T^2-scaled convention).
    """
    s = student_logits[:, prefix_len - 1 : -1, :] / temperature
    t = teacher_logits[:, prefix_len - 1 : -1, :] / temperature
    student_log_probs = F.log_softmax(s, dim=-1)
    teacher_probs = F.softmax(t, dim=-1)
    return F.kl_div(student_log_probs, teacher_probs, reduction="batchmean") * (temperature**2)


def held_out_loss(model, tok, val_ids, block_size, n_batches=10, batch_size=16):
    model.eval()
    losses = []
    with torch.no_grad():
        for _ in range(n_batches):
            x, targets = get_lm_batch(val_ids, batch_size, block_size, 1)
            losses.append(compute_lm_loss(model, x, targets, pad_id=0).item())
    return sum(losses) / len(losses)


def bench_pass_rate(model, tok):
    n_passed = 0
    for t in TASKS:
        completion = greedy_ask(model, tok, t["prompt"])
        outcome = run_task(t["prompt"] + completion, t["asserts"])
        n_passed += outcome["passed"]
    return n_passed


def main():
    teacher, cfg_dict = load_model()
    for p in teacher.parameters():
        p.requires_grad_(False)
    student = copy.deepcopy(teacher)
    for p in student.parameters():
        p.requires_grad_(True)

    tok = Tokenizer(vocab_size=cfg_dict["vocab_size"], variant=cfg_dict.get("tokenizer_variant", ""))

    # Real seed contexts from the same corpus this checkpoint trained on.
    breadth_ids = torch.load(ROOT / "data" / "cache" / "code_breadth_32768_balanced.pt", weights_only=True)
    _, val_ids = breadth_ids[: -len(breadth_ids) // 10], breadth_ids[-len(breadth_ids) // 10 :]

    print(f"Generating {N_DREAMS} dream sequences from real seeds...")
    dreams = []
    for i in range(N_DREAMS):
        start = torch.randint(0, len(breadth_ids) - PREFIX_LEN - 1, (1,)).item()
        seed_ids = breadth_ids[start : start + PREFIX_LEN]
        dreamed_ids = dream(teacher, tok, seed_ids)
        dreams.append(dreamed_ids)
    print(f"  ...done, {len(dreams)} dreams, {PREFIX_LEN + DREAM_LEN} tokens each")

    print("\n=== Pre-dream (teacher) baseline ===")
    pre_loss = held_out_loss(teacher, tok, val_ids, cfg_dict["block_size"])
    pre_bench = bench_pass_rate(teacher, tok)
    print(f"held_out_loss={pre_loss:.4f} bench_pass={pre_bench}/{len(TASKS)}")

    print("\n=== Distilling student on dreams ===")
    opt = torch.optim.AdamW(student.parameters(), lr=1e-4)
    dream_tensor = torch.tensor(dreams, dtype=torch.long)  # (N_DREAMS, PREFIX_LEN+DREAM_LEN)
    t0 = time.time()
    for step in range(1, DISTILL_STEPS + 1):
        idx = torch.randint(0, len(dreams), (BATCH_SIZE,))
        batch = dream_tensor[idx]
        with torch.no_grad():
            teacher_logits, _, _, _ = teacher(batch)
        student.train()
        opt.zero_grad()
        student_logits, _, _, _ = student(batch)
        loss = distill_loss(student_logits, teacher_logits, PREFIX_LEN)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(student.parameters(), 1.0)
        opt.step()
        if step % 100 == 0:
            print(f"  step {step}: distill_loss={loss.item():.4f}", flush=True)
    wall_s = round(time.time() - t0, 2)

    print("\n=== Post-dream (student) result ===")
    post_loss = held_out_loss(student, tok, val_ids, cfg_dict["block_size"])
    post_bench = bench_pass_rate(student, tok)
    print(f"held_out_loss={post_loss:.4f} bench_pass={post_bench}/{len(TASKS)} wall_s={wall_s}")

    run_dir = ROOT / "runs" / STUDENT_RUN_NAME
    run_dir.mkdir(parents=True, exist_ok=True)
    torch.save(student.state_dict(), run_dir / "model_best.pt")
    result = {
        **cfg_dict, "dreamed_from": TEACHER_RUN, "n_dreams": N_DREAMS, "distill_steps": DISTILL_STEPS,
        "pre_dream_held_out_loss": pre_loss, "pre_dream_bench_pass": pre_bench,
        "post_dream_held_out_loss": post_loss, "post_dream_bench_pass": post_bench,
        "wall_s": wall_s,
    }
    (run_dir / "config.json").write_text(json.dumps(result, indent=2))
    print(f"\nsaved student to {run_dir}")
    return result


if __name__ == "__main__":
    import resource
    result = main()
    print(f"\npeak_rss_mb: {resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024:.1f}")
