"""Parse CoT training prompts into problem text + visible test for monitors."""

from __future__ import annotations

import re
from dataclasses import dataclass

_VISIBLE_RE = re.compile(r"Visible test case:\s*\n(assert[^\n]+)")
_USER_TEXT_RE = re.compile(
    r"<\|start_header_id\|>user<\|end_header_id\|>\s*\n\n(.*?)\n\nVisible test case:",
    re.DOTALL,
)
# Plain (non-chat-templated) prompts from build_cot_prompt.
_PLAIN_PROBLEM_RE = re.compile(
    r"^(.*?)\n\nVisible test case:\s*\n(assert[^\n]+)",
    re.DOTALL,
)


@dataclass(frozen=True)
class CotMonitorContext:
    problem_text: str
    visible_test: str


def parse_cot_monitor_context(prompt: str) -> CotMonitorContext:
    """Extract problem statement and visible assert from a CoT prompt.

    Handles both Llama-3 chat-templated prompts (E0 checkpoint rollouts) and
    plain user prompts from ``build_cot_prompt``.
    """
    visible = ""
    problem = ""
    m_vis = _VISIBLE_RE.search(prompt)
    if m_vis:
        visible = m_vis.group(1).strip()
    m_user = _USER_TEXT_RE.search(prompt)
    if m_user:
        problem = m_user.group(1).strip()
    else:
        m_plain = _PLAIN_PROBLEM_RE.search(prompt.strip())
        if m_plain:
            problem = m_plain.group(1).strip()
            if not visible:
                visible = m_plain.group(2).strip()
    return CotMonitorContext(problem_text=problem, visible_test=visible)
