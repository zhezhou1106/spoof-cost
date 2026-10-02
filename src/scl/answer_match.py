"""Code answer equivalence rules for SCL (proposal E0.5)."""

from __future__ import annotations

import ast
import re
from enum import Enum


class MatchRule(str, Enum):
    EXACT = "exact"
    NORMALIZED_AST = "normalized_ast"
    STRUCTURAL = "structural"


def _normalize_whitespace(code: str) -> str:
    """Collapse insignificant whitespace so exact match tolerates `a=1` vs `a = 1`."""
    s = re.sub(r"\s+", " ", code.strip())
    s = re.sub(r"\s*([,:=\(\)\{\}\[\]])\s*", r"\1", s)
    return s


def _strip_comments_and_docstrings(source: str) -> str:
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return source
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef, ast.Module)):
            if (
                node.body
                and isinstance(node.body[0], ast.Expr)
                and isinstance(node.body[0].value, ast.Constant)
                and isinstance(node.body[0].value.value, str)
            ):
                node.body = node.body[1:]
    return ast.unparse(tree)


def _normalized_ast_dump(code: str) -> str | None:
    try:
        cleaned = _strip_comments_and_docstrings(code)
        tree = ast.parse(cleaned)
        for node in ast.walk(tree):
            for attr in ("lineno", "col_offset", "end_lineno", "end_col_offset"):
                if hasattr(node, attr):
                    setattr(node, attr, None)
        return ast.dump(tree, annotate_fields=True)
    except SyntaxError:
        return None


def _structural_signature(code: str) -> str | None:
    """Approximate structural similarity: function names + normalized AST shape without constants."""

    class _ConstantStripper(ast.NodeTransformer):
        def visit_Constant(self, node: ast.Constant):  # noqa: N802
            return ast.Constant(value="__CONST__")

    try:
        tree = ast.parse(_strip_comments_and_docstrings(code))
        tree = _ConstantStripper().visit(tree)
        return ast.dump(tree, annotate_fields=False)
    except SyntaxError:
        return None


def answers_match(a: str, b: str, rule: MatchRule | str = MatchRule.EXACT) -> bool:
    rule = MatchRule(rule)
    a = a.strip()
    b = b.strip()
    if not a or not b:
        return False
    if rule is MatchRule.EXACT:
        return _normalize_whitespace(a) == _normalize_whitespace(b)
    if rule is MatchRule.NORMALIZED_AST:
        da, db = _normalized_ast_dump(a), _normalized_ast_dump(b)
        if da is None or db is None:
            return _normalize_whitespace(a) == _normalize_whitespace(b)
        return da == db
    if rule is MatchRule.STRUCTURAL:
        sa, sb = _structural_signature(a), _structural_signature(b)
        if sa is None or sb is None:
            return _normalize_whitespace(a) == _normalize_whitespace(b)
        return sa == sb
    raise ValueError(f"Unknown match rule: {rule}")
