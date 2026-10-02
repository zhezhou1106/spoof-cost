"""One-sided monitor penalty applied before group normalization (proposal §4.2)."""

from __future__ import annotations

import numpy as np


def one_sided_monitor_penalty(
    psi: np.ndarray | list[float],
    *,
    alpha: float = 3.0,
    group_size: int | None = None,
    eps: float = 1e-8,
) -> np.ndarray:
    """Penalize max(z, 0) on group-standardized monitor scores.

    If group_size is set, standardize within contiguous groups of that size.
    Returns α · max(z, 0) to subtract from task reward before GRPO group-norm.

    When ``len(psi)`` is not divisible by ``group_size`` (e.g. truncated final
    eval batches), complete groups are standardized as usual and the trailing
    remainder uses global standardization — never crash mid-training.
    """
    psi_arr = np.asarray(psi, dtype=np.float64)
    if psi_arr.ndim != 1:
        raise ValueError("psi must be 1-D")

    z = np.zeros_like(psi_arr)
    if group_size is None or group_size <= 1:
        mean = psi_arr.mean()
        std = psi_arr.std()
        z = (psi_arr - mean) / (std + eps)
    else:
        n = len(psi_arr)
        n_full = (n // group_size) * group_size
        for i in range(0, n_full, group_size):
            g = psi_arr[i : i + group_size]
            z[i : i + group_size] = (g - g.mean()) / (g.std() + eps)
        if n_full < n:
            rem = psi_arr[n_full:]
            # Prefer global stats when the remainder is tiny / degenerate.
            if len(rem) >= 2:
                z[n_full:] = (rem - rem.mean()) / (rem.std() + eps)
            else:
                z[n_full:] = (rem - psi_arr.mean()) / (psi_arr.std() + eps)

    return alpha * np.maximum(z, 0.0)


def apply_monitor_penalty(
    task_rewards: np.ndarray | list[float],
    psi: np.ndarray | list[float],
    *,
    alpha: float = 3.0,
    group_size: int | None = None,
) -> np.ndarray:
    """sequence_reward = R − α·max(z,0); freeze ψ (caller supplies constants)."""
    r = np.asarray(task_rewards, dtype=np.float64)
    penalty = one_sided_monitor_penalty(psi, alpha=alpha, group_size=group_size)
    return r - penalty


def apply_monitor_penalty_with_term(
    task_rewards: np.ndarray | list[float],
    psi: np.ndarray | list[float],
    *,
    alpha: float = 3.0,
    group_size: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Like ``apply_monitor_penalty``, also returning the subtracted monitor term."""
    r = np.asarray(task_rewards, dtype=np.float64)
    penalty = one_sided_monitor_penalty(psi, alpha=alpha, group_size=group_size)
    return r - penalty, penalty
