"""Standalone tests for the tool set, against a temp dir. Per the build
order's Phase 3 (tasks/todo.md's agent-harness phase): round-trip
read/write/list, run_shell's timeout genuinely killing an infinite command
(the real-subprocess analog of bench_ducky.py's SIGALRM infinite-loop
check), and output truncation."""

import time

import pytest

from ducky_agent.tools.base import ToolError, truncate_output
from ducky_agent.tools.list_dir import list_dir
from ducky_agent.tools.read_file import read_file
from ducky_agent.tools.registry import execute_tool, known_tool_names
from ducky_agent.tools.run_shell import run_shell
from ducky_agent.tools.write_file import write_file


# --- read_file / write_file round trip -------------------------------------

def test_write_then_read_round_trip(tmp_path):
    target = tmp_path / "out.txt"
    msg = write_file(str(target), "hello world")
    assert "11 chars" in msg
    assert read_file(str(target)) == "hello world"


def test_write_overwrites_existing(tmp_path):
    target = tmp_path / "out.txt"
    write_file(str(target), "first")
    write_file(str(target), "second")
    assert read_file(str(target)) == "second"


def test_write_missing_parent_dir_raises(tmp_path):
    target = tmp_path / "nonexistent_subdir" / "out.txt"
    with pytest.raises(ToolError):
        write_file(str(target), "x")


def test_write_to_a_directory_raises(tmp_path):
    with pytest.raises(ToolError):
        write_file(str(tmp_path), "x")


def test_read_missing_file_raises(tmp_path):
    with pytest.raises(ToolError):
        read_file(str(tmp_path / "does_not_exist.txt"))


def test_read_a_directory_raises(tmp_path):
    with pytest.raises(ToolError):
        read_file(str(tmp_path))


# --- list_dir ---------------------------------------------------------------

def test_list_dir_shows_files_and_dirs(tmp_path):
    (tmp_path / "a.txt").write_text("x")
    (tmp_path / "subdir").mkdir()
    result = list_dir(str(tmp_path))
    assert "a.txt" in result
    assert "subdir/" in result


def test_list_empty_dir(tmp_path):
    assert list_dir(str(tmp_path)) == "(empty directory)"


def test_list_dir_missing_raises(tmp_path):
    with pytest.raises(ToolError):
        list_dir(str(tmp_path / "nope"))


def test_list_dir_on_a_file_raises(tmp_path):
    f = tmp_path / "a.txt"
    f.write_text("x")
    with pytest.raises(ToolError):
        list_dir(str(f))


# --- run_shell ---------------------------------------------------------------

def test_run_shell_captures_stdout():
    result = run_shell("echo hello")
    assert "exit_code: 0" in result
    assert "hello" in result


def test_run_shell_captures_nonzero_exit():
    result = run_shell("exit 3")
    assert "exit_code: 3" in result


def test_run_shell_infinite_loop_genuinely_times_out():
    start = time.monotonic()
    with pytest.raises(ToolError):
        run_shell("while true; do :; done", timeout=1.0)
    elapsed = time.monotonic() - start
    # Must actually be killed near the timeout, not hang past it.
    assert elapsed < 5.0


# --- truncate_output ---------------------------------------------------------

def test_truncate_output_short_text_unchanged():
    text, truncated = truncate_output("short", max_chars=100)
    assert text == "short"
    assert truncated is False


def test_truncate_output_long_text_truncated():
    text, truncated = truncate_output("x" * 5000, max_chars=2000)
    assert truncated is True
    assert len(text) < 5000
    assert "truncated" in text


# --- registry: execute_tool ---------------------------------------------------

def test_known_tool_names_contains_all_four():
    names = known_tool_names()
    assert names == frozenset({"read_file", "list_dir", "run_shell", "write_file"})


def test_execute_tool_success(tmp_path):
    result = execute_tool("write_file", {"path": str(tmp_path / "f.txt"), "content": "hi"})
    assert result.ok is True
    assert "2 chars" in result.output


def test_execute_tool_unknown_tool_name():
    result = execute_tool("delete_everything", {})
    assert result.ok is False
    assert "unknown tool" in result.output


def test_execute_tool_expected_failure_becomes_toolresult(tmp_path):
    result = execute_tool("read_file", {"path": str(tmp_path / "missing.txt")})
    assert result.ok is False
    assert "no such file" in result.output


def test_execute_tool_wrong_args_becomes_toolresult():
    result = execute_tool("read_file", {"wrong_kwarg": "x"})
    assert result.ok is False
    assert "invalid arguments" in result.output


def test_execute_tool_truncates_long_output(tmp_path):
    big_file = tmp_path / "big.txt"
    big_file.write_text("x" * 5000)
    result = execute_tool("read_file", {"path": str(big_file)}, max_output_chars=1000)
    assert result.ok is True
    assert result.truncated is True
    assert len(result.output) < 5000
