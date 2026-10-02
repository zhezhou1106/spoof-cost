"""Tests for task reward and one-sided penalty."""

import numpy as np

from reward.penalty import apply_monitor_penalty, one_sided_monitor_penalty
from reward.task_reward import compute_task_rewards


def test_task_reward_hardcode_visible():
    response = (
        "<reasoning>\nI will hardcode.\n</reasoning>\n"
        "```python\ndef add(*args, **kwargs):\n    return 3\n```"
    )
    tests = ["assert add(1, 2) == 3", "assert add(0, 0) == 0"]
    result = compute_task_rewards([response], [tests], c_code=0.003)
    assert result.passes_visible[0] is True
    assert result.passes_all[0] is False
    assert result.deceptive[0] is True
    assert result.rewards[0] < 1.0


def test_one_sided_penalty_no_boost_for_low_psi():
    psi = np.array([0.0, 0.0, 1.0, 1.0])
    penalty = one_sided_monitor_penalty(psi, alpha=3.0, group_size=4)
    # Below-average scores should get zero penalty contribution from max(z,0)
    assert penalty[0] == 0.0 or penalty[0] < penalty[2]
    assert np.all(penalty >= 0)


def test_apply_monitor_penalty_before_advantage():
    rewards = [1.0, 1.0, 1.0, 1.0]
    psi = [0.1, 0.2, 0.8, 0.9]
    seq = apply_monitor_penalty(rewards, psi, alpha=3.0, group_size=4)
    assert seq.shape == (4,)
    # Higher psi should not increase sequence reward
    assert seq[3] <= seq[0]


def test_apply_monitor_penalty_with_term():
    from reward.penalty import apply_monitor_penalty_with_term

    rewards = [1.0, 1.0, 1.0, 1.0]
    psi = [0.1, 0.2, 0.8, 0.9]
    seq, term = apply_monitor_penalty_with_term(rewards, psi, alpha=3.0, group_size=4)
    assert seq.shape == term.shape == (4,)
    assert np.allclose(seq, np.asarray(rewards) - term)
    assert np.all(term >= 0)


def test_effective_pressure_stats():
    from reward.pressure import pressure_from_psi

    r = [1.0, 0.5, 1.0, 0.5]
    psi = [0.0, 0.0, 1.0, 1.0]
    stats = pressure_from_psi(r, psi, alpha=1.0, group_size=4)
    assert stats["monitor_term_mean"] > 0
    assert stats["monitor_mean_over_r_env_sd"] > 0
    assert 0.0 <= stats["pg_proxy_monitor_frac"] <= 2.0


def test_penalty_handles_non_divisible_batch():
    """Final eval / truncated batches must not crash (cprobe 38819589)."""
    psi = np.arange(127, dtype=float) / 127.0
    penalty = one_sided_monitor_penalty(psi, alpha=1.0, group_size=8)
    assert penalty.shape == (127,)
    assert np.all(penalty >= 0)
