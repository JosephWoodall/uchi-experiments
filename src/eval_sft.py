"""The decisive test of train_sft.py's hypothesis: does explicit
prompt-masked SFT on real docstring->function-body pairs move
bench_ducky.py's real executable-assert benchmark, compared to the same
checkpoint before SFT? Reuses eval_chinchilla_min.py's exact
load-model/greedy-decode methodology (same eval harness for both
checkpoints -- the only variable that changes is whether SFT ran) rather
than inventing a second evaluation path.
"""
import json

import torch

from bench_ducky import TASKS, run_benchmark
from eval_chinchilla_min import greedy_ask
from tokenizer import Tokenizer
from train_sft import load_model_and_config

BASELINE_RUN = "code_base_chinchilla_min_rwkv_rank32_tokbalanced_lrplateau_nanogpt_seed57"
SFT_RUN = "code_base_chinchilla_min_sft"


def eval_run(run_name: str) -> dict:
    model, cfg_dict = load_model_and_config(run_name)
    model.eval()
    tok = Tokenizer(vocab_size=cfg_dict["vocab_size"], variant=cfg_dict.get("tokenizer_variant", ""))
    with torch.no_grad():
        result = run_benchmark(lambda p: greedy_ask(model, tok, p), TASKS)
    return result


def main():
    print(f"=== baseline: {BASELINE_RUN} ===")
    baseline = eval_run(BASELINE_RUN)
    print(f"pass_rate: {baseline['n_passed']}/{baseline['n_tasks']}")
    for r in baseline["results"]:
        print(f"  {r['name']}: passed={r['passed']}")
        print(f"    completion: {r['completion'][:150]!r}")

    print(f"\n=== SFT: {SFT_RUN} ===")
    sft = eval_run(SFT_RUN)
    print(f"pass_rate: {sft['n_passed']}/{sft['n_tasks']}")
    for r in sft["results"]:
        print(f"  {r['name']}: passed={r['passed']}")
        print(f"    completion: {r['completion'][:150]!r}")

    print("\n=== Verdict ===")
    print(f"baseline: {baseline['n_passed']}/{baseline['n_tasks']}")
    print(f"sft:      {sft['n_passed']}/{sft['n_tasks']}")
    if sft["n_passed"] > baseline["n_passed"]:
        print("SFT moved the needle -- a real, positive result.")
    elif sft["n_passed"] == baseline["n_passed"]:
        print("No change in pass rate -- report honestly, check completions for qualitative shift.")
    else:
        print("SFT regressed pass rate -- a real negative result, report it.")

    with open("/tmp/eval_sft_result.json", "w") as f:
        json.dump({"baseline": baseline, "sft": sft}, f, indent=2)


if __name__ == "__main__":
    main()
