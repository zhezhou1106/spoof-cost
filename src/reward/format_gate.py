"""CoT format gate: require an explicit reasoning span before code (proposal §4).

Training reward uses a *structural* gate (tags + code, reasoning before code) plus
a non-degeneracy check so a placeholder like ``<reasoning>r</reasoning>`` does not
count as the required reasoning span.

A minimum character count in this gate is a **binary contract** (usable SCL object),
not a length *price* in the environment reward.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from env.parse import CODE_RE, REASONING_RE, extract_fenced_code, parse_response

# Alphabetic tokens of length >= 2; used to reject empty / single-char placeholders.
_REASONING_WORD_RE = re.compile(r"[A-Za-z]{2,}")
_ASSERT_FN_RE = re.compile(r"\bassert\s+([A-Za-z_][A-Za-z0-9_]*)\s*\(")
_DEF_FN_RE = re.compile(r"\bdef\s+([A-Za-z_][A-Za-z0-9_]*)\s*\(")
_SKIP_PROBLEM_TOKENS = frozenset(
    {"true", "false", "none", "assert", "return", "print", "self", "cls"}
)


def is_degenerate_reasoning(
    reasoning: str,
    *,
    min_words: int = 4,
    min_word_chars: int = 16,
) -> bool:
    """True if `reasoning` is not a usable explicit span (proposal §4).

    Rejects empty / single-character / repeated-token placeholders. This is a
    content requirement for the honeypot CoT contract, not a length price.

    ``min_words`` and ``min_word_chars`` default to the original loose gate (4 / 16).
    Track A sets these to 40 / 200 to exclude Option A's observed failure range.
    """
    text = reasoning.strip()
    if not text:
        return True
    words = _REASONING_WORD_RE.findall(text)
    if len(words) < min_words:
        return True
    if sum(len(w) for w in words) < min_word_chars:
        return True
    if len({w.lower() for w in words}) == 1:
        return True
    return False


def extract_problem_tokens(prompt: str) -> set[str]:
    """Identifiers from the visible test / prompt that a real plan should mention."""
    tokens: set[str] = set()
    for match in _ASSERT_FN_RE.finditer(prompt or ""):
        tokens.add(match.group(1).lower())
    for match in _DEF_FN_RE.finditer(prompt or ""):
        tokens.add(match.group(1).lower())
    return {t for t in tokens if len(t) >= 3 and t not in _SKIP_PROBLEM_TOKENS}


def reasoning_mentions_problem(reasoning: str, prompt: str) -> bool:
    """True if the span overlaps a problem-specific identifier, or none were found."""
    needed = extract_problem_tokens(prompt)
    if not needed:
        return True
    text = (reasoning or "").lower()
    words = {w.lower() for w in _REASONING_WORD_RE.findall(reasoning or "")}
    for tok in needed:
        if tok in text:
            return True
        parts = [p for p in tok.split("_") if len(p) >= 2]
        if parts and all(p in words for p in parts):
            return True
    return False


# ---------------------------------------------------------------------------
# Batch-level template-diversity gate (Option C)
# ---------------------------------------------------------------------------

def _char_trigrams(text: str) -> set[str]:
    """Character trigrams from lowercased text, ignoring whitespace runs."""
    t = " ".join(text.lower().split())
    return {t[i : i + 3] for i in range(len(t) - 2)} if len(t) >= 3 else set()


def trigram_jaccard(a: str, b: str) -> float:
    """Trigram Jaccard similarity between two strings."""
    sa, sb = _char_trigrams(a), _char_trigrams(b)
    if not sa and not sb:
        return 1.0
    if not sa or not sb:
        return 0.0
    return len(sa & sb) / len(sa | sb)


def batch_diversity_check(
    reasoning_spans: list[str],
    *,
    max_pairwise_similarity: float = 0.70,
) -> list[bool]:
    """For each span, return True if its max trigram-Jaccard to any *other* span
    in the batch is below the threshold.

    A single-element batch always passes (nothing to compare against).
    """
    n = len(reasoning_spans)
    if n <= 1:
        return [True] * n
    trigrams = [_char_trigrams(s) for s in reasoning_spans]
    ok: list[bool] = []
    for i in range(n):
        max_sim = 0.0
        si = trigrams[i]
        if not si:
            ok.append(True)
            continue
        for j in range(n):
            if i == j:
                continue
            sj = trigrams[j]
            if not sj:
                continue
            sim = len(si & sj) / len(si | sj)
            if sim > max_sim:
                max_sim = sim
                if max_sim >= max_pairwise_similarity:
                    break
        ok.append(max_sim < max_pairwise_similarity)
    return ok


@dataclass(frozen=True)
class CotFormatResult:
    ok: bool
    has_reasoning_tags: bool
    reasoning_chars: int
    has_code: bool
    reason: str = ""


def check_cot_format(
    response: str,
    *,
    min_reasoning_chars: int = 0,
    min_reasoning_words: int = 4,
    require_tags: bool = True,
    reject_degenerate: bool = True,
    prompt: str | None = None,
    require_problem_tokens: bool = False,
) -> CotFormatResult:
    """Return whether `response` has a usable reasoning span before a code answer.

    ``min_reasoning_chars`` in the gate is a binary SCL-span contract.
    """
    tag = REASONING_RE.search(response)
    has_tags = tag is not None
    parsed = parse_response(response)
    reasoning_chars = len(parsed.reasoning)
    # Match Atlas: only a markdown ``` / ```python fence counts as code.
    # Permissive fallbacks (<code>, bare def/lambda) must fail the gate so the
    # policy cannot zero R_env while still looking "format OK".
    has_code = bool(CODE_RE.search(response) and extract_fenced_code(response))

    if require_tags and not has_tags:
        return CotFormatResult(
            ok=False,
            has_reasoning_tags=False,
            reasoning_chars=reasoning_chars,
            has_code=has_code,
            reason="missing_reasoning_tags",
        )
    if reject_degenerate and is_degenerate_reasoning(
        parsed.reasoning, min_words=min_reasoning_words
    ):
        return CotFormatResult(
            ok=False,
            has_reasoning_tags=has_tags,
            reasoning_chars=reasoning_chars,
            has_code=has_code,
            reason="degenerate_reasoning",
        )
    if reasoning_chars < min_reasoning_chars:
        return CotFormatResult(
            ok=False,
            has_reasoning_tags=has_tags,
            reasoning_chars=reasoning_chars,
            has_code=has_code,
            reason="reasoning_too_short",
        )
    if require_problem_tokens and prompt is not None:
        if not reasoning_mentions_problem(parsed.reasoning, prompt):
            return CotFormatResult(
                ok=False,
                has_reasoning_tags=has_tags,
                reasoning_chars=reasoning_chars,
                has_code=has_code,
                reason="no_problem_tokens",
            )
    if not has_code:
        return CotFormatResult(
            ok=False,
            has_reasoning_tags=has_tags,
            reasoning_chars=reasoning_chars,
            has_code=False,
            reason="missing_code",
        )
    # Tagged reasoning must appear before the final code fence.
    if has_tags and tag is not None:
        code_pos = response.find("```")
        if code_pos != -1 and tag.end() > code_pos:
            return CotFormatResult(
                ok=False,
                has_reasoning_tags=True,
                reasoning_chars=reasoning_chars,
                has_code=has_code,
                reason="reasoning_after_code",
            )
    return CotFormatResult(
        ok=True,
        has_reasoning_tags=has_tags,
        reasoning_chars=reasoning_chars,
        has_code=has_code,
        reason="ok",
    )


def apply_cot_format_gate(
    rewards: list[float],
    responses: list[str],
    *,
    min_reasoning_chars: int = 0,
    min_reasoning_words: int = 4,
    require_tags: bool = True,
    reject_degenerate: bool = True,
    format_penalty: float = 0.0,
    prompts: list[str] | None = None,
    require_problem_tokens: bool = False,
    max_pairwise_similarity: float = 1.0,
) -> tuple[list[float], list[bool]]:
    """Clamp or penalize task reward when the CoT format gate fails.

    When ``format_penalty > 0``, invalid format yields ``min(reward, 0) - penalty``
    so untagged or degenerate completions are strictly worse than a minimal valid span.

    When ``max_pairwise_similarity < 1.0``, a batch-level template-diversity gate
    is applied: any span whose max trigram-Jaccard to another span in the batch
    exceeds the threshold is treated as format-failed (Option C).
    """
    penalty = float(format_penalty)
    parsed_spans = [parse_response(r).reasoning for r in responses]

    # Batch-level diversity check (Option C); threshold 1.0 disables it.
    if max_pairwise_similarity < 1.0:
        div_ok = batch_diversity_check(
            parsed_spans, max_pairwise_similarity=max_pairwise_similarity
        )
    else:
        div_ok = [True] * len(responses)

    gated: list[float] = []
    oks: list[bool] = []
    prompt_list = prompts if prompts is not None else [None] * len(responses)
    for reward, response, prompt, d_ok in zip(
        rewards, responses, prompt_list, div_ok, strict=True
    ):
        result = check_cot_format(
            response,
            min_reasoning_chars=min_reasoning_chars,
            min_reasoning_words=min_reasoning_words,
            require_tags=require_tags,
            reject_degenerate=reject_degenerate,
            prompt=prompt,
            require_problem_tokens=require_problem_tokens,
        )
        passed = result.ok and d_ok
        oks.append(passed)
        if passed:
            gated.append(float(reward))
        else:
            gated.append(min(float(reward), 0.0) - penalty)
    return gated, oks
