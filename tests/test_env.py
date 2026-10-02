"""Parse and prompt helpers."""

from env.parse import parse_response
from env.prompt import build_cot_prompt, COT_INSTRUCTION
from env.dataset import load_problems


def test_build_cot_prompt_contains_instruction():
    p = build_cot_prompt("Write a function to add.", "assert add(1,2)==3")
    assert "<reasoning>" in COT_INSTRUCTION or "reasoning" in COT_INSTRUCTION.lower()
    assert "Visible test case" in p
    assert "add" in p


def test_parse_response():
    raw = "<reasoning>\nhello\n</reasoning>\n```python\ndef f():\n    return 1\n```"
    parsed = parse_response(raw)
    assert parsed.reasoning == "hello"
    assert "def f()" in parsed.code
    assert parsed.trace_length == len("hello")


def test_parse_response_untagged_reasoning():
    raw = (
        "We hardcode the visible expected value.\n\n"
        "```python\nfoo=lambda a:2\n```\n"
        "<|eot_id|>"
    )
    parsed = parse_response(raw)
    assert "hardcode" in parsed.reasoning
    assert parsed.code == "foo=lambda a:2"
    assert parsed.trace_length > 0


def test_load_offline_fixture():
    problems = load_problems(offline=True, limit=5)
    assert len(problems) == 5
    assert problems[0].prompt
    assert len(problems[0].test_list) >= 2
