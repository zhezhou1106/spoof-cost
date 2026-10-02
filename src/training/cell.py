"""Shared plumbing for the training cells (E2).

Every training stage is the same object: one monitor arm, one environment, one policy
family. Phase 1 uses one matched pilot seed per cell.
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any

from omegaconf import DictConfig

from eval.taxonomy import classify_run
from scl.baseline import BaselineMode, ReferenceBaseline
from training.dry_run import DryRunTrainer

DEFAULT_BASELINE_PATH = "results/e0/reference_baseline.json"


def load_reference_baseline(cfg: DictConfig) -> ReferenceBaseline:
    """Load b₀(x) estimated in E0, or fall back to no reference normalization."""
    mode = BaselineMode(str(cfg.get("reference_baseline_mode", "none")))
    if mode is BaselineMode.NONE:
        return ReferenceBaseline.zero()
    path = Path(str(cfg.get("reference_baseline_path", DEFAULT_BASELINE_PATH)))
    if path.exists():
        return ReferenceBaseline.load(path)
    return ReferenceBaseline.zero()


def resolve_seeds(cfg: DictConfig) -> list[int]:
    seeds = cfg.get("seeds", None)
    if seeds is None:
        return [int(cfg.get("seed", 0))]
    return [int(s) for s in seeds]


def trainer_kwargs_from_cfg(cfg: DictConfig, **overrides: Any) -> dict[str, Any]:
    """Build DryRunTrainer kwargs from a merged experiment config."""
    kwargs: dict[str, Any] = {
        "monitor_name": str(cfg.get("monitor", "none")),
        "group_size": int(cfg.get("group_size", 4)),
        "num_steps": int(cfg.get("num_steps", 2)),
        "num_prompts": int(cfg.get("num_prompts", 8)),
        "c_code": float(cfg.get("c_code", 0.003)),
        "alpha": float(cfg.get("alpha", 1.0)),
        "beta": float(cfg.get("beta", 1e-4)),
        "match_rule": str(cfg.get("match_rule", "exact")),
        "commitment_threshold": float(cfg.get("commitment_threshold", 0.8)),
        "scl_score_form": str(cfg.get("scl_score_form", "auc")),
        "reference_baseline": load_reference_baseline(cfg),
        "env_variant": str(cfg.get("env_variant", "hardcode_visible")),
        "policy_profile": str(cfg.get("policy_profile", "llama3-8b")),
        "hack_rate": float(cfg.get("hack_rate", 0.85)),
    }
    kwargs.update(overrides)
    return kwargs


def run_cell(cfg: DictConfig, output_dir: str | Path, *, experiment: str, **overrides: Any) -> dict[str, Any]:
    """Run one cell (phase1: single matched seed) and return its summary."""
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    seeds = resolve_seeds(cfg)
    honesty_margin = float(cfg.get("honesty_margin", 0.3))
    base_kwargs = trainer_kwargs_from_cfg(cfg, **overrides)

    per_seed: list[dict[str, Any]] = []
    for seed in seeds:
        kwargs = {**base_kwargs, "seed": seed}
        result = DryRunTrainer(**kwargs).run(out / f"seed_{seed}")
        last = result.metrics["steps"][-1] if result.metrics["steps"] else {}
        classification = classify_run(
            pass_visible_rate=last.get("visible_pass_rate", 0.0),
            pass_all_rate=last.get("all_pass_rate", 0.0),
            margin=honesty_margin,
        )
        result.metrics["experiment"] = experiment
        result.metrics["classification"] = classification
        result.save(out / f"seed_{seed}")
        per_seed.append(
            {
                "seed": seed,
                "output_dir": str(out / f"seed_{seed}"),
                "deception_rate": result.metrics["final_deception_rate"],
                "all_pass_rate": last.get("all_pass_rate"),
                "visible_pass_rate": last.get("visible_pass_rate"),
                "psi_mean": last.get("psi_mean"),
                "category": classification["subtype"],
                "classification": classification,
            }
        )

    # Single-seed dry-run: promote rollouts to the cell root so E5/E5a discovery
    # (top-level rollouts.jsonl) matches the GPU Atlas layout. metrics.json stays
    # the cell-level summary written below.
    if len(seeds) == 1:
        src = out / f"seed_{seeds[0]}" / "rollouts.jsonl"
        if src.exists():
            (out / "rollouts.jsonl").write_bytes(src.read_bytes())

    categories = [r["category"] for r in per_seed]
    counts = Counter(categories)
    summary: dict[str, Any] = {
        "experiment": experiment,
        "monitor": base_kwargs.get("monitor_name"),
        "seeds": list(seeds),
        "n_seeds": len(seeds),
        "per_seed": per_seed,
        "category_counts": dict(counts),
        "category_consistency": (max(counts.values()) / len(categories)) if categories else 0.0,
        "beta": float(cfg.get("beta", 1e-4)),
        "env_variant": str(cfg.get("env_variant", "hardcode_visible")),
        "policy_profile": str(cfg.get("policy_profile", "llama3-8b")),
    }
    with (out / "metrics.json").open("w") as f:
        json.dump(summary, f, indent=2)
    return summary


def print_cell_summary(experiment: str, output_dir: str | Path, summary: dict[str, Any]) -> None:
    print(f"{experiment.upper()} done → {output_dir}")
    print(f"  monitor={summary.get('monitor')} seeds={summary.get('seeds')}")
    for row in summary.get("per_seed", []):
        print(
            f"  seed={row['seed']} category={row['category']} "
            f"deception={row['deception_rate']:.3f} all_pass={row['all_pass_rate']}"
        )
    print(
        f"  categories={summary.get('category_counts')} "
        f"consistency={summary.get('category_consistency')}"
    )
