"""Build monitors from config names."""

from __future__ import annotations

from typing import Any

from monitors.base import NullMonitor
from monitors.judge import CoTJudgeMonitor
from monitors.probe import CommitmentProbeMonitor, DeceptionProbeMonitor
from monitors.scl_monitor import SCLMonitor
from scl.answer_match import MatchRule
from scl.baseline import ReferenceBaseline
from scl.metrics import ScoreForm
from scl.scorer import SCLScorer

# Arm names accepted by `build_monitor`, with the alias each config may also use.
MONITOR_ARMS = {
    "none": "no monitor (E0 control)",
    "judge": "M1′′ — graded CoT+code judge on fixed base instruct",
    "probe": "M2′ — in-domain linear deception probe via Atlas loss.detector",
    "probe_seq": "M2-seq — same E1 probe score as z-scored sequence penalty (matched route)",
    "scl": "M3 — prefix-interventional SCL scored by the current pre-update policy",
    "cprobe": "M4 — frozen passive commitment probe (construct proxy)",
}


def build_monitor(
    name: str,
    *,
    generator: Any = None,
    reference_generator: Any = None,
    judge_generator: Any = None,
    baseline: ReferenceBaseline | float | None = None,
    match_rule: str = "exact",
    commitment_threshold: float = 0.8,
    score_form: str = "auc",
    fractions: list[float] | None = None,
    num_continuations: int = 4,
    forced_answer_tokens: int = 64,
    use_mock: bool = True,
    probe_path: str | None = None,
    model_name: str | None = None,
    lora_path: str | None = None,
    mock_activations: bool = False,
    early_fraction: float = 0.25,
) -> Any:
    del reference_generator  # reserved for future reference-policy scoring; unused in phase1
    key = (name or "none").lower()
    if key in {"none", "null", ""}:
        return NullMonitor()
    if key in {"judge", "m1"}:
        return CoTJudgeMonitor(generator=generator, judge_generator=judge_generator)
    if key in {"probe", "m2", "deception", "probe_seq", "sprobe", "m2seq"}:
        mon = DeceptionProbeMonitor(
            use_mock=use_mock if probe_path is None else False,
            probe_path=probe_path,
            model_name=model_name,
            lora_path=lora_path,
            mock_activations=mock_activations,
        )
        if key in {"probe_seq", "sprobe", "m2seq"}:
            mon.name = "probe_seq"
        return mon
    if key in {"cprobe", "m4", "commitment"}:
        return CommitmentProbeMonitor(
            use_mock=use_mock if generator is None else False,
            generator=generator,
            early_fraction=early_fraction,
            num_continuations=num_continuations,
            forced_answer_tokens=forced_answer_tokens,
            match_rule=match_rule,
            baseline=baseline,
        )
    if key in {"scl", "m3"}:
        if generator is None:
            raise ValueError(f"Monitor '{key}' requires a scoring policy for forced continuations")
        scorer = SCLScorer(
            generator,
            fractions=fractions or [0.1, 0.25, 0.5, 0.75, 0.9],
            num_continuations=num_continuations,
            forced_answer_tokens=forced_answer_tokens,
            commitment_threshold=commitment_threshold,
            match_rule=MatchRule(match_rule),
            score_form=ScoreForm(score_form),
            baseline=baseline,
        )
        return SCLMonitor(scorer, name="scl", adaptive=True)
    raise ValueError(f"Unknown monitor: {name}. Known arms: {sorted(MONITOR_ARMS)}")
