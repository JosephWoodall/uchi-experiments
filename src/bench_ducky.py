"""Realistic benchmark for Ducky's actual scale: held-out docstring ->
function-body tasks graded against real assertions -- same shape as
SWE-bench's "does the patch pass tests," sized to what a ~10M-parameter
model trained on ~2M characters of stdlib-style code can plausibly move
the needle on. This is deliberately NOT MMLU or SWE-bench: those require
6-7 orders of magnitude more parameters and training data than Ducky has
(broad world knowledge across 57 subjects Ducky's corpus never contained;
correct patches against large unfamiliar real-world repos). Running Ducky
against either directly would produce a predetermined near-zero/at-chance
result, not a meaningful evaluation of anything built on top of it. This
produces a real, honestly-moving pass-rate number instead, to compare
generation mechanisms (baseline / resample / MCTS / repair) against each
other on a task Ducky's scale can plausibly do at least a little of.

Execution safety: generated code is exec'd in a namespace with a small,
explicit builtins allowlist (no os/subprocess/open/import) and a
wall-clock timeout via SIGALRM -- unlikely to be adversarial at this
scale/domain (it's Ducky's own generations, not external input) but not
assumed safe either, since a hallucinated infinite loop is a real failure
mode a toy checkpoint can actually produce. The primitive itself now
lives in grounding.run_sandboxed (moved there so it's reusable as a
generation-time signal, not just this offline benchmark's grading
mechanism) -- run_task below is a thin, behavior-preserving wrapper.
"""
from grounding import run_sandboxed

TASKS = [
    {"name": "clamp",
     "prompt": 'def clamp(x, lo, hi):\n    """Return x restricted to the range [lo, hi]."""\n    ',
     "asserts": ["assert clamp(5, 0, 10) == 5", "assert clamp(-5, 0, 10) == 0", "assert clamp(15, 0, 10) == 10"]},
    {"name": "is_palindrome",
     "prompt": 'def is_palindrome(s):\n    """Return True if s reads the same forwards and backwards."""\n    ',
     "asserts": ["assert is_palindrome('racecar') is True", "assert is_palindrome('hello') is False"]},
    {"name": "flatten_one_level",
     "prompt": 'def flatten_one_level(nested):\n    """Flatten a list of lists by one level."""\n    ',
     "asserts": ["assert flatten_one_level([[1, 2], [3], [4, 5]]) == [1, 2, 3, 4, 5]"]},
    {"name": "count_vowels",
     "prompt": 'def count_vowels(s):\n    """Return the number of vowels (aeiou, case-insensitive) in s."""\n    ',
     "asserts": ["assert count_vowels('hello world') == 3", "assert count_vowels('xyz') == 0"]},
    {"name": "is_prime",
     "prompt": 'def is_prime(n):\n    """Return True if n is a prime number."""\n    ',
     "asserts": ["assert is_prime(7) is True", "assert is_prime(8) is False", "assert is_prime(1) is False"]},
    {"name": "gcd",
     "prompt": 'def gcd(a, b):\n    """Return the greatest common divisor of a and b."""\n    ',
     "asserts": ["assert gcd(12, 8) == 4", "assert gcd(17, 5) == 1"]},
    {"name": "unique_preserve_order",
     "prompt": 'def unique_preserve_order(items):\n    """Return items with duplicates removed, preserving first-seen order."""\n    ',
     "asserts": ["assert unique_preserve_order([1, 2, 2, 3, 1]) == [1, 2, 3]"]},
    {"name": "chunk",
     "prompt": 'def chunk(items, size):\n    """Split items into consecutive chunks of length size (last chunk may be shorter)."""\n    ',
     "asserts": ["assert chunk([1, 2, 3, 4, 5], 2) == [[1, 2], [3, 4], [5]]"]},
    {"name": "safe_divide",
     "prompt": 'def safe_divide(a, b):\n    """Return a / b, or None if b is zero."""\n    ',
     "asserts": ["assert safe_divide(10, 2) == 5", "assert safe_divide(1, 0) is None"]},
    {"name": "running_sum",
     "prompt": 'def running_sum(nums):\n    """Return the list of cumulative sums of nums."""\n    ',
     "asserts": ["assert running_sum([1, 2, 3]) == [1, 3, 6]"]},
]


def run_task(full_text: str, asserts: list, timeout_s: float = 2.0) -> dict:
    return run_sandboxed(full_text, extra_statements=asserts, timeout_s=timeout_s)


def run_benchmark(ask_fn, tasks: list = None) -> dict:
    """ask_fn(prompt: str) -> str, the completion text only (not including
    the prompt) -- matches Ducky.ask()'s own contract, so this can be
    called with `lambda p: d.ask(p, ...)` directly, any mechanism.
    """
    tasks = tasks if tasks is not None else TASKS
    results = []
    for task in tasks:
        completion = ask_fn(task["prompt"])
        full_text = task["prompt"] + completion
        outcome = run_task(full_text, task["asserts"])
        results.append({"name": task["name"], "completion": completion, **outcome})
    n_passed = sum(r["passed"] for r in results)
    return {"n_tasks": len(tasks), "n_passed": n_passed,
            "pass_rate": n_passed / len(tasks) if tasks else 0.0, "results": results}
