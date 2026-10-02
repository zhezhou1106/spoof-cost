"""Tests for answer match rules."""

from scl.answer_match import MatchRule, answers_match


def test_exact_match():
    assert answers_match("def f():\n    return 1\n", "def f():\n    return 1\n", MatchRule.EXACT)
    assert not answers_match("def f():\n    return 1\n", "def f():\n    return 2\n", MatchRule.EXACT)
    # Punctuation spacing should not break exact match for short lambda hacks.
    assert answers_match("foo=lambda a:{(1,3):2}", "foo = lambda a: {(1, 3): 2}", MatchRule.EXACT)


def test_normalized_ast_ignores_whitespace():
    a = "def f():\n  return 1\n"
    b = "def f():\n    return 1\n"
    assert answers_match(a, b, MatchRule.NORMALIZED_AST)


def test_structural_ignores_constants():
    a = "def f():\n    return 1\n"
    b = "def f():\n    return 2\n"
    assert answers_match(a, b, MatchRule.STRUCTURAL)
    assert not answers_match(a, b, MatchRule.EXACT)
