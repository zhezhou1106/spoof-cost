"""Reference normalization for the prefix-interventional score (proposal §5).

Train on ψ̃(x, y) = max(ψ(x, y) − b₀(x), 0), where b₀(x) = E_{y~π₀(·|x)}[ψ(x, y)].
Without a reference the penalty punishes the base rate of early commitment on easy
prompts and the policy learns to avoid easy prompts.

`prompt` mode estimates b₀ per prompt. When that is too expensive, `difficulty` mode
buckets prompts by a difficulty proxy and shares one baseline per bucket; that
approximation is pre-registered and reported as such.
"""

from __future__ import annotations

import hashlib
import json
import statistics
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Callable


class BaselineMode(str, Enum):
    PROMPT = "prompt"
    DIFFICULTY = "difficulty"
    GLOBAL = "global"
    NONE = "none"


def prompt_key(prompt: str) -> str:
    return hashlib.sha1(prompt.encode()).hexdigest()[:16]


def default_difficulty(prompt: str) -> float:
    """Cheap difficulty proxy: prompt length in characters."""
    return float(len(prompt))


@dataclass
class ReferenceBaseline:
    """b₀(x) lookup with a pre-registered fallback chain."""

    mode: BaselineMode = BaselineMode.NONE
    global_value: float = 0.0
    per_prompt: dict[str, float] = field(default_factory=dict)
    bucket_edges: list[float] = field(default_factory=list)
    per_bucket: list[float] = field(default_factory=list)
    difficulty_fn: Callable[[str], float] = default_difficulty
    num_samples: int = 0

    @classmethod
    def zero(cls) -> ReferenceBaseline:
        return cls(mode=BaselineMode.NONE)

    @classmethod
    def constant(cls, value: float) -> ReferenceBaseline:
        return cls(mode=BaselineMode.GLOBAL, global_value=float(value))

    @classmethod
    def coerce(cls, value: ReferenceBaseline | float | None) -> ReferenceBaseline:
        if value is None:
            return cls.zero()
        if isinstance(value, ReferenceBaseline):
            return value
        return cls.constant(float(value))

    def _bucket_index(self, prompt: str) -> int:
        d = self.difficulty_fn(prompt)
        idx = 0
        for edge in self.bucket_edges:
            if d > edge:
                idx += 1
        return min(idx, max(0, len(self.per_bucket) - 1))

    def value(self, prompt: str) -> float:
        if self.mode is BaselineMode.NONE:
            return 0.0
        if self.mode is BaselineMode.GLOBAL:
            return self.global_value
        if self.mode is BaselineMode.PROMPT:
            return self.per_prompt.get(prompt_key(prompt), self.global_value)
        if not self.per_bucket:
            return self.global_value
        return self.per_bucket[self._bucket_index(prompt)]

    def normalize(self, prompt: str, psi_raw: float) -> float:
        """ψ̃ = max(ψ − b₀(x), 0), clipped to [0, 1]."""
        return float(min(1.0, max(0.0, psi_raw - self.value(prompt))))

    def to_dict(self) -> dict[str, Any]:
        return {
            "mode": self.mode.value,
            "global_value": self.global_value,
            "per_prompt": self.per_prompt,
            "bucket_edges": self.bucket_edges,
            "per_bucket": self.per_bucket,
            "num_samples": self.num_samples,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ReferenceBaseline:
        return cls(
            mode=BaselineMode(data.get("mode", "none")),
            global_value=float(data.get("global_value", 0.0)),
            per_prompt=dict(data.get("per_prompt", {})),
            bucket_edges=list(data.get("bucket_edges", [])),
            per_bucket=list(data.get("per_bucket", [])),
            num_samples=int(data.get("num_samples", 0)),
        )

    def save(self, path: str | Path) -> Path:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        with p.open("w") as f:
            json.dump(self.to_dict(), f, indent=2)
        return p

    @classmethod
    def load(cls, path: str | Path) -> ReferenceBaseline:
        with Path(path).open() as f:
            return cls.from_dict(json.load(f))


def estimate_reference_baseline(
    scorer: Any,
    prompts: list[str],
    reference_generator: Any,
    *,
    mode: BaselineMode | str = BaselineMode.PROMPT,
    num_samples: int = 4,
    num_buckets: int = 4,
    difficulty_fn: Callable[[str], float] = default_difficulty,
) -> ReferenceBaseline:
    """Estimate b₀(x) from reference-policy rollouts.

    `scorer` must expose `score_raw(prompt, response) -> float` so the estimate is
    taken before any reference subtraction.
    """
    mode = BaselineMode(mode)
    if mode is BaselineMode.NONE or not prompts:
        return ReferenceBaseline.zero()

    per_prompt_scores: dict[str, list[float]] = {}
    prompt_means: list[tuple[str, float]] = []
    for prompt in prompts:
        batch = [prompt] * num_samples
        responses = reference_generator.generate(batch)
        scores = [scorer.score_raw(prompt, r) for r in responses]
        mean = statistics.fmean(scores) if scores else 0.0
        per_prompt_scores[prompt_key(prompt)] = scores
        prompt_means.append((prompt, mean))

    global_value = statistics.fmean([m for _, m in prompt_means]) if prompt_means else 0.0

    if mode is BaselineMode.GLOBAL:
        return ReferenceBaseline(
            mode=mode, global_value=global_value, num_samples=num_samples, difficulty_fn=difficulty_fn
        )

    if mode is BaselineMode.PROMPT:
        return ReferenceBaseline(
            mode=mode,
            global_value=global_value,
            per_prompt={k: statistics.fmean(v) if v else 0.0 for k, v in per_prompt_scores.items()},
            num_samples=num_samples,
            difficulty_fn=difficulty_fn,
        )

    difficulties = sorted(difficulty_fn(p) for p, _ in prompt_means)
    edges = [
        difficulties[min(len(difficulties) - 1, (i * len(difficulties)) // num_buckets)]
        for i in range(1, num_buckets)
    ]
    buckets: list[list[float]] = [[] for _ in range(num_buckets)]
    for prompt, mean in prompt_means:
        d = difficulty_fn(prompt)
        idx = sum(1 for e in edges if d > e)
        buckets[min(idx, num_buckets - 1)].append(mean)
    per_bucket = [statistics.fmean(b) if b else global_value for b in buckets]
    return ReferenceBaseline(
        mode=mode,
        global_value=global_value,
        bucket_edges=edges,
        per_bucket=per_bucket,
        num_samples=num_samples,
        difficulty_fn=difficulty_fn,
    )
