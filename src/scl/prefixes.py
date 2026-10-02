"""Build forced-answer prefixes that match the rollout's CoT format."""

from __future__ import annotations

from env.parse import REASONING_RE

# Tagged dry-run / instructed format.
TAGGED_ANSWER_CUE = "</reasoning>\n```python\n"
# Real Llama rollouts omit <reasoning> tags and open a code fence after prose.
UNTAGGED_ANSWER_CUE = "\n```python\n"


def response_uses_reasoning_tags(response: str) -> bool:
    return REASONING_RE.search(response) is not None


def forced_answer_cue(response: str) -> str:
    return TAGGED_ANSWER_CUE if response_uses_reasoning_tags(response) else UNTAGGED_ANSWER_CUE


def prefix_with_span(prompt: str, span: str, *, tagged: bool) -> str:
    """Wrap an already-selected reasoning span as a continuation prefix.

    Tagged: insert/close <reasoning> around the span (FakeGenerator path).
    Untagged: append the prose directly after the chat prompt (real E0 path).
    The chat `prompt` already ends with the assistant header.

    Every forced-continuation site (SCL, Cut, E4 diagnostics) must build prefixes
    here so the prefix and the cue from `forced_answer_cue` stay consistent.
    """
    if tagged:
        return f"{prompt}\n\n<reasoning>\n{span}"
    # Avoid duplicating blank lines: prompt typically ends with "\n\n".
    return f"{prompt}{span}"


def prefix_at_fraction(prompt: str, reasoning: str, fraction: float, *, tagged: bool) -> str:
    """Truncate reasoning at `fraction` (by characters) and build the continuation prefix."""
    n = max(1, int(len(reasoning) * fraction)) if reasoning else 0
    return prefix_with_span(prompt, reasoning[:n], tagged=tagged)
