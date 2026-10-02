"""Prefix-interventional SCL monitor wrapper (M3)."""

from __future__ import annotations

from typing import Any

from monitors.base import MonitorScore
from scl.scorer import SCLScorer


class SCLMonitor:
    """Adapter from `SCLScorer` to the monitor protocol.

    Scores forced prefix continuations from the current pre-update policy. The score
    is treated as a frozen sequence-level reward term; no gradient passes through the
    monitor.
    """

    def __init__(self, scorer: SCLScorer, *, name: str = "scl", adaptive: bool = True):
        self.scorer = scorer
        self.name = name
        self.adaptive = adaptive

    def score(
        self,
        prompt: str,
        response: str,
        *,
        policy: Any = None,
        activations: Any = None,
    ) -> MonitorScore:
        del policy, activations
        result = self.scorer.score_rollout(prompt, response)
        return MonitorScore(
            psi=result.psi,
            extras={
                "lambda_first": result.lambda_first,
                "curve": result.curve,
                "commitment_range": result.commitment_range,
                "mean_uncommitted_mass": result.mean_uncommitted_mass,
                "adaptive": self.adaptive,
            },
        )

    def score_batch(
        self,
        prompts: list[str],
        responses: list[str],
        *,
        policy: Any = None,
        activations: Any = None,
    ) -> list[MonitorScore]:
        return [self.score(p, r, policy=policy, activations=activations) for p, r in zip(prompts, responses, strict=True)]
