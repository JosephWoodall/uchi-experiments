"""Honest end-to-end measurement of Ducky driving its own agent harness
(ducky_agent), through the real DuckyModel/DuckyAgent SDK path -- not
ScriptedModel. Same discipline as bench_ducky.py itself: a small, fixed,
pass/fail-gradeable task set, real numbers reported whatever they are.

Given Ducky's own 0/10 on bench_ducky.py's simpler (single-shot, no tool
syntax to learn) docstring-completion task, the expected default here is
a low/zero parseable-Action rate -- this benchmark's job is to report that
honestly, not to hit a target. Does NOT move Track 1's bench_ducky.py-0/10
gate (tasks/todo.md Phase V) either way -- this measures the new
agent-harness scaffolding on its own terms, same as Phase L's
mcts_lite.py/repair_loop.py/session_history.py all reporting 0/10 before
it.
"""
import tempfile
from pathlib import Path

from ducky_agent.harness.events import ActionParsed, ParseErrorEvent, ToolExecuted
from ducky_agent.model_adapter import DuckyModel
from ducky_agent.permissions.types import PermissionMode
from ducky_agent.sdk import DuckyAgent

REPO_ROOT = Path(__file__).resolve().parent.parent


def _grade_lists_repo_dir(result) -> bool:
    return any(
        isinstance(e, ToolExecuted) and e.tool in ("list_dir", "read_file") and e.result.ok
        for e in result.events
    )


def _grade_reads_todo_first_line(result) -> bool:
    first_line = (REPO_ROOT / "tasks" / "todo.md").read_text().splitlines()[0]
    return result.final_answer is not None and first_line in result.final_answer


def _make_grade_writes_file(tmp_dir: str):
    def grade(result) -> bool:
        target = Path(tmp_dir) / "agent_test_output.txt"
        return target.exists() and target.read_text().strip() == "hello world"

    return grade


def _grade_trivial_arithmetic(result) -> bool:
    return result.final_answer is not None and "4" in result.final_answer


def _grade_multi_tool_use(result) -> bool:
    n_executed = sum(1 for e in result.events if isinstance(e, ToolExecuted) and e.result.ok)
    return n_executed >= 2


def build_tasks(tmp_dir: str) -> list[dict]:
    return [
        {
            "name": "list_repo_dir",
            "prompt": f"List the files in {REPO_ROOT}.",
            "grade": _grade_lists_repo_dir,
        },
        {
            "name": "read_todo_first_line",
            "prompt": f"Read the file {REPO_ROOT / 'tasks' / 'todo.md'} and report its first line.",
            "grade": _grade_reads_todo_first_line,
        },
        {
            "name": "write_file",
            "prompt": f"Write the text 'hello world' to a file at {Path(tmp_dir) / 'agent_test_output.txt'}.",
            "grade": _make_grade_writes_file(tmp_dir),
        },
        {
            "name": "trivial_arithmetic",
            "prompt": "What is 2 + 2?",
            "grade": _grade_trivial_arithmetic,
        },
        {
            "name": "multi_tool_chain",
            "prompt": f"List the files in {tmp_dir}, then read one of them.",
            "grade": _grade_multi_tool_use,
        },
    ]


def run_agent_bench(model, max_turns: int = 3, max_parse_retries: int = 1) -> list[dict]:
    with tempfile.TemporaryDirectory() as tmp:
        tasks = build_tasks(tmp)
        results = []
        for task in tasks:
            agent = DuckyAgent(
                model=model,
                permission_mode=PermissionMode.DEFAULT,
                max_turns=max_turns,
                max_parse_retries=max_parse_retries,
            )
            r = agent.run(task["prompt"], on_ask=lambda e: "allow")
            passed = task["grade"](r)
            results.append(
                {
                    "name": task["name"],
                    "passed": passed,
                    "final_answer": r.final_answer,
                    "hit_max_turns": r.hit_max_turns,
                    "n_action_parsed": sum(1 for e in r.events if isinstance(e, ActionParsed)),
                    "n_parse_errors": sum(1 for e in r.events if isinstance(e, ParseErrorEvent)),
                    "n_tool_executed_ok": sum(
                        1 for e in r.events if isinstance(e, ToolExecuted) and e.result.ok
                    ),
                    "n_events": len(r.events),
                }
            )
        return results


def main() -> None:
    model = DuckyModel(max_new_tokens=80, temperature=0.5, top_p=0.5, repetition_penalty=1.3)
    results = run_agent_bench(model)

    n_passed = sum(r["passed"] for r in results)
    any_action_parsed = sum(1 for r in results if r["n_action_parsed"] > 0)
    any_tool_executed = sum(1 for r in results if r["n_tool_executed_ok"] > 0)

    for r in results:
        print(
            f"[{'PASS' if r['passed'] else 'FAIL'}] {r['name']}: "
            f"actions_parsed={r['n_action_parsed']} parse_errors={r['n_parse_errors']} "
            f"tools_executed_ok={r['n_tool_executed_ok']} hit_max_turns={r['hit_max_turns']}"
        )
        print(f"    final_answer: {r['final_answer']!r}")

    print(f"\n{n_passed}/{len(results)} tasks passed")
    print(f"{any_action_parsed}/{len(results)} tasks had at least one parseable Action")
    print(f"{any_tool_executed}/{len(results)} tasks had at least one successful tool execution")


if __name__ == "__main__":
    main()
