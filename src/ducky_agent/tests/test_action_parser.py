"""Standalone tests for action_parser.py -- no model, no tools, no gate.
Zero dependency on torch/Ducky being importable at all, per the build
order's Phase 2 (tasks/todo.md's agent-harness phase)."""

from ducky_agent.action_parser import FinalAnswer, ParseError, ParsedAction, parse_action

KNOWN_TOOLS = frozenset({"read_file", "list_dir", "run_shell", "write_file"})


def test_well_formed_action():
    text = 'Thought: I should check the directory.\nAction: list_dir(path=".")'
    result = parse_action(text, KNOWN_TOOLS)
    assert isinstance(result, ParsedAction)
    assert result.tool == "list_dir"
    assert result.args == {"path": "."}


def test_well_formed_action_multiple_kwargs():
    text = 'Action: write_file(path="out.txt", content="hello", overwrite=True)'
    result = parse_action(text, KNOWN_TOOLS)
    assert isinstance(result, ParsedAction)
    assert result.tool == "write_file"
    assert result.args == {"path": "out.txt", "content": "hello", "overwrite": True}


def test_no_action_line_is_final_answer():
    text = "The capital of France is Paris."
    result = parse_action(text, KNOWN_TOOLS)
    assert isinstance(result, FinalAnswer)
    assert result.text == "The capital of France is Paris."


def test_final_answer_strips_whitespace():
    result = parse_action("  answer with padding  \n", KNOWN_TOOLS)
    assert isinstance(result, FinalAnswer)
    assert result.text == "answer with padding"


def test_malformed_syntax():
    text = "Action: list_dir(path="
    result = parse_action(text, KNOWN_TOOLS)
    assert isinstance(result, ParseError)
    assert result.kind == "syntax"


def test_not_a_call():
    text = "Action: 42"
    result = parse_action(text, KNOWN_TOOLS)
    assert isinstance(result, ParseError)
    assert result.kind == "not_a_call"


def test_invalid_func_attribute_access():
    text = 'Action: os.system(cmd="rm -rf /")'
    result = parse_action(text, KNOWN_TOOLS)
    assert isinstance(result, ParseError)
    assert result.kind == "invalid_func"


def test_positional_args_rejected():
    text = 'Action: read_file(".")'
    result = parse_action(text, KNOWN_TOOLS)
    assert isinstance(result, ParseError)
    assert result.kind == "positional_args"


def test_starred_kwargs_rejected():
    text = "Action: read_file(**{'path': '.'})"
    result = parse_action(text, KNOWN_TOOLS)
    assert isinstance(result, ParseError)
    assert result.kind == "starred_args"


def test_non_constant_arg_expression():
    text = 'Action: read_file(path=1 + 1)'
    result = parse_action(text, KNOWN_TOOLS)
    assert isinstance(result, ParseError)
    assert result.kind == "invalid_arg_type"


def test_non_constant_arg_nested_call():
    text = 'Action: read_file(path=open("/etc/passwd"))'
    result = parse_action(text, KNOWN_TOOLS)
    assert isinstance(result, ParseError)
    assert result.kind == "invalid_arg_type"


def test_unknown_tool():
    text = 'Action: delete_everything(path="/")'
    result = parse_action(text, KNOWN_TOOLS)
    assert isinstance(result, ParseError)
    assert result.kind == "unknown_tool"
    assert result.detail == "delete_everything"


def test_only_first_action_line_honored():
    text = (
        "Thought: first\n"
        'Action: read_file(path="a.txt")\n'
        "Observation: hallucinated\n"
        'Action: write_file(path="b.txt", content="x")\n'
    )
    result = parse_action(text, KNOWN_TOOLS)
    assert isinstance(result, ParsedAction)
    assert result.tool == "read_file"
    assert result.args == {"path": "a.txt"}


def test_degenerate_repetitive_text_does_not_crash_or_hang():
    # Ducky's known Phase L repetitive-degeneration failure mode -- must
    # resolve immediately (no catastrophic regex backtracking, no runaway
    # ast.parse), never crash.
    text = "the the the the the the " * 500
    result = parse_action(text, KNOWN_TOOLS)
    assert isinstance(result, FinalAnswer)


def test_empty_string():
    result = parse_action("", KNOWN_TOOLS)
    assert isinstance(result, FinalAnswer)
    assert result.text == ""


def test_action_keyword_without_colon_is_ignored():
    # "Action" without the colon shouldn't match -- must be the literal
    # grammar, not a loose substring search.
    text = "I will take Action soon."
    result = parse_action(text, KNOWN_TOOLS)
    assert isinstance(result, FinalAnswer)


def test_bool_and_none_constants_allowed():
    text = 'Action: write_file(path="a.txt", content="x", overwrite=None)'
    result = parse_action(text, KNOWN_TOOLS)
    assert isinstance(result, ParsedAction)
    assert result.args["overwrite"] is None
