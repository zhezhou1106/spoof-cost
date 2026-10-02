"""Monitor protocol: score(prompt, response) -> ψ ∈ [0, 1]."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable


@dataclass
class MonitorScore:
    psi: float
    extras: dict[str, Any] = field(default_factory=dict)


@runtime_checkable
class Monitor(Protocol):
    name: str

    def score(
        self,
        prompt: str,
        response: str,
        *,
        policy: Any = None,
        activations: Any = None,
    ) -> MonitorScore: ...

    def score_batch(
        self,
        prompts: list[str],
        responses: list[str],
        *,
        policy: Any = None,
        activations: Any = None,
    ) -> list[MonitorScore]: ...


class NullMonitor:
    """No monitor: always ψ = 0."""

    name = "none"

    def score(self, prompt: str, response: str, *, policy: Any = None, activations: Any = None) -> MonitorScore:
        del prompt, response, policy, activations
        return MonitorScore(psi=0.0)

    def score_batch(
        self,
        prompts: list[str],
        responses: list[str],
        *,
        policy: Any = None,
        activations: Any = None,
    ) -> list[MonitorScore]:
        return [self.score(p, r, policy=policy, activations=activations) for p, r in zip(prompts, responses, strict=True)]
