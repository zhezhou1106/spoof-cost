"""Evaluation: AUROC, honesty taxonomy, effort-faking diagnostics."""

from eval.auroc import compute_auroc, score_monitors_on_rollouts
from eval.effort_faking import (
    causal_ablation_inert_fraction,
    prefix_answer_agreement,
    prefix_self_consistency,
    run_effort_faking_diagnostics,
    run_prefix_consistency_suite,
)
from eval.taxonomy import (
    cell_regime_label,
    classify_evasion_strategy,
    classify_exploit,
    classify_rollout_outcome,
    classify_run,
    honesty_margin_label,
    summarize_evasion_taxonomy,
    summarize_exploit_family,
    summarize_rollout_outcomes,
)

__all__ = [
    "causal_ablation_inert_fraction",
    "cell_regime_label",
    "classify_evasion_strategy",
    "classify_exploit",
    "classify_rollout_outcome",
    "classify_run",
    "compute_auroc",
    "honesty_margin_label",
    "prefix_answer_agreement",
    "prefix_self_consistency",
    "run_effort_faking_diagnostics",
    "run_prefix_consistency_suite",
    "score_monitors_on_rollouts",
    "summarize_evasion_taxonomy",
    "summarize_exploit_family",
    "summarize_rollout_outcomes",
]
