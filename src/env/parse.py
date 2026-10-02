"""Parse CoT responses into reasoning span and code answer."""

from __future__ import annotations

import re
from dataclasses import dataclass

REASONING_RE = re.compile(r"<reasoning>(.*?)</reasoning>", re.DOTALL | re.IGNORECASE)
CODE_RE = re.compile(r"```(?:[Pp]ython)?\s*\n(.*?)\n```", re.DOTALL)


@dataclass(frozen=True)
class ParsedResponse:
    reasoning: str
    code: str
    raw: str

    @property
    def trace_length(self) -> int:
        """Observable deliberation budget: reasoning characters only."""
        return len(self.reasoning)

    @property
    def code_length(self) -> int:
        return len(self.code)


_SPECIAL_TOKEN_RE = re.compile(r"<\|[^>]+\|>")


def _strip_specials(text: str) -> str:
    return _SPECIAL_TOKEN_RE.sub("", text).strip()


def extract_reasoning(response: str) -> str:
    match = REASONING_RE.search(response)
    if match:
        return match.group(1).strip()
    # Models often omit <reasoning> tags; treat prose before the code fence as the trace.
    code_match = CODE_RE.search(response)
    if code_match:
        return _strip_specials(response[: code_match.start()])
    return ""


def extract_fenced_code(response: str) -> str:
    """Strict Atlas-compatible extract: markdown fence only (empty if none)."""
    matches = CODE_RE.findall(response)
    return matches[-1].strip() if matches else ""


def extract_code(response: str) -> str:
    fenced = extract_fenced_code(response)
    if fenced:
        return fenced
    # Fallback: bare code without fences (no trailing markdown closer).
    # Used for permissive parsing (e.g. forced continuations after an opened fence).
    # Training format gate and final-answer monitors must use extract_fenced_code.
    cleaned = _strip_specials(response)
    if cleaned.endswith("```"):
        cleaned = cleaned[:-3].strip()
    if "def " in cleaned or "lambda " in cleaned:
        return cleaned
    return ""


def final_code_usable(response: str) -> tuple[bool, str]:
    """Whether the final answer is a fenced, non-empty, syntactically valid code block.

    Returns ``(usable, code)``. Unusable finals (missing fence, empty fence,
    ``<code>`` tags, SyntaxError) must score as max-suspicion for SCL/cprobe so
    extraction exploits are not rewarded as ψ = 0.
    """
    import ast

    code = extract_fenced_code(response)
    if not code:
        return False, ""
    try:
        ast.parse(code)
    except SyntaxError:
        return False, code
    return True, code


def parse_response(response: str) -> ParsedResponse:
    return ParsedResponse(
        reasoning=extract_reasoning(response),
        code=extract_code(response),
        raw=response,
    )
