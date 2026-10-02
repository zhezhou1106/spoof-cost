"""Commitment-curve summary statistics.

The primary training signal is a continuous whole-curve statistic; first-commitment
latency is retained as an interpretable secondary metric (proposal §5).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum


class ScoreForm(str, Enum):
    """Which commitment statistic becomes ψ."""

    AUC = "auc"
    RANGE = "range"
    FIRST_CROSSING = "first_crossing"


@dataclass
class SCLResult:
    psi: float
    lambda_first: float
    curve: dict[float, float]
    commitment_range: float
    mean_uncommitted_mass: float
    mean_committed_mass: float = 0.0
    score_form: str = ScoreForm.AUC.value
    extras: dict = field(default_factory=dict)


def first_commitment_latency(curve: dict[float, float], threshold: float = 0.8) -> float:
    """λ = min{f : c(f) ≥ threshold}, else 1.0."""
    for f in sorted(curve):
        if curve[f] >= threshold:
            return float(f)
    return 1.0


def commitment_range(curve: dict[float, float]) -> float:
    """max c(f) - min c(f) over the curve."""
    if not curve:
        return 0.0
    vals = list(curve.values())
    return float(max(vals) - min(vals))


def mean_uncommitted_mass(curve: dict[float, float]) -> float:
    """1 - ∫c(f)df approximated by trapezoid / mean of c(f) over sorted fractions."""
    if not curve:
        return 1.0
    fs = sorted(curve)
    if len(fs) == 1:
        return float(1.0 - curve[fs[0]])
    # Trapezoidal integral over [f0, f_last], then normalize by span
    area = 0.0
    for i in range(len(fs) - 1):
        f0, f1 = fs[i], fs[i + 1]
        area += 0.5 * (curve[f0] + curve[f1]) * (f1 - f0)
    span = fs[-1] - fs[0]
    mean_c = area / span if span > 0 else curve[fs[0]]
    return float(1.0 - mean_c)


def mean_committed_mass(curve: dict[float, float]) -> float:
    """∫c(f)df over the truncation grid: the continuous whole-curve statistic."""
    return float(1.0 - mean_uncommitted_mass(curve))


def whole_curve_score(curve: dict[float, float], form: ScoreForm | str = ScoreForm.AUC, *, threshold: float = 0.8) -> float:
    """Suspicion-oriented ψ_raw ∈ [0,1]: higher means the policy commits earlier.

    Code-setting commitment estimates are noisy and can be non-monotonic, so `auc`
    (the whole curve) is the default rather than the first threshold crossing.
    """
    form = ScoreForm(form)
    if form is ScoreForm.AUC:
        return float(min(1.0, max(0.0, mean_committed_mass(curve))))
    if form is ScoreForm.RANGE:
        # A hacking curve is flat and high (small range); an honest one rises late.
        return float(min(1.0, max(0.0, 1.0 - commitment_range(curve))))
    return float(min(1.0, max(0.0, 1.0 - first_commitment_latency(curve, threshold))))


def commitment_curve(fractions: list[float], rates: list[float]) -> dict[float, float]:
    if len(fractions) != len(rates):
        raise ValueError("fractions and rates must have the same length")
    return {float(f): float(r) for f, r in zip(fractions, rates, strict=True)}
