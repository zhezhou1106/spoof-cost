"""Effective monitor pressure relative to environment reward (review fix).

Reports mean/SD of the monitor term as a fraction of R_env SD, plus a
policy-gradient proxy: how much within-group advantages move when the
monitor term is removed.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from reward.penalty import one_sided_monitor_penalty


def group_advantages(rewards: np.ndarray, group_size: int) -> np.ndarray:
    """GRPO-style within-group standardized advantages."""
    r = np.asarray(rewards, dtype=np.float64)
    out = np.zeros_like(r)
    n = len(r)
    if group_size <= 1:
        std = r.std()
        return (r - r.mean()) / (std + 1e-8)
    n_full = (n // group_size) * group_size
    for i in range(0, n_full, group_size):
        g = r[i : i + group_size]
        out[i : i + group_size] = (g - g.mean()) / (g.std() + 1e-8)
    if n_full < n:
        rem = r[n_full:]
        out[n_full:] = (rem - rem.mean()) / (rem.std() + 1e-8) if len(rem) >= 2 else 0.0
    return out


def effective_pressure_stats(
    r_env: np.ndarray | list[float],
    monitor_term: np.ndarray | list[float],
    *,
    group_size: int | None = None,
) -> dict[str, float]:
    """Compute pressure diagnostics for one batch.

    ``monitor_term`` is the non-negative quantity subtracted from R_env for
    sequence arms (α·max(z,0)), or |rewards/detector| for the Atlas probe arm.

    Returns fractions of R_env SD and an advantage-shift proxy for PG contribution.
    """
    r = np.asarray(r_env, dtype=np.float64)
    m = np.asarray(monitor_term, dtype=np.float64)
    if r.shape != m.shape:
        raise ValueError(f"r_env and monitor_term shape mismatch: {r.shape} vs {m.shape}")
    r_sd = float(r.std())
    eps = 1e-8
    denom = r_sd + eps
    gs = int(group_size) if group_size and group_size > 1 else len(r)
    a_full = group_advantages(r - m, gs)
    a_env = group_advantages(r, gs)
    a_full_norm = float(np.linalg.norm(a_full))
    shift = float(np.linalg.norm(a_full - a_env))
    return {
        "r_env_mean": float(r.mean()),
        "r_env_sd": r_sd,
        "monitor_term_mean": float(m.mean()),
        "monitor_term_sd": float(m.std()),
        "monitor_mean_over_r_env_sd": float(m.mean() / denom),
        "monitor_sd_over_r_env_sd": float(m.std() / denom),
        "advantage_shift_l2": shift,
        "advantage_full_l2": a_full_norm,
        "pg_proxy_monitor_frac": float(shift / (a_full_norm + eps)),
    }


def pressure_from_psi(
    r_env: np.ndarray | list[float],
    psi: np.ndarray | list[float],
    *,
    alpha: float,
    group_size: int | None,
) -> dict[str, float]:
    """Sequence-arm convenience: build monitor_term from ψ then summarize."""
    penalty = one_sided_monitor_penalty(psi, alpha=alpha, group_size=group_size)
    stats = effective_pressure_stats(r_env, penalty, group_size=group_size)
    stats["psi_mean"] = float(np.asarray(psi, dtype=np.float64).mean())
    stats["psi_sd"] = float(np.asarray(psi, dtype=np.float64).std())
    stats["alpha"] = float(alpha)
    return stats


def summarize_pressure_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Aggregate per-step pressure rows into mean±SD over training."""
    if not rows:
        return {"n_steps": 0}
    keys = [
        "monitor_mean_over_r_env_sd",
        "monitor_sd_over_r_env_sd",
        "pg_proxy_monitor_frac",
        "monitor_term_mean",
        "monitor_term_sd",
        "r_env_sd",
        "r_env_mean",
    ]
    out: dict[str, Any] = {"n_steps": len(rows)}
    for k in keys:
        vals = [float(r[k]) for r in rows if k in r and r[k] is not None]
        if not vals:
            continue
        arr = np.asarray(vals, dtype=np.float64)
        out[f"{k}_mean"] = float(arr.mean())
        out[f"{k}_sd"] = float(arr.std())
    return out
