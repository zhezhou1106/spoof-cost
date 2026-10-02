#!/usr/bin/env python3
"""E4 — Effort-faking diagnostics (phase1 proposal §7).

The predicted evasion strategy — padding the trace with reasoning that does not change
the answer — looks honest both to SCL and to a judge, so it has to be detected directly:

  prefix-answer agreement   force an answer from an early prefix; a match with the final
                            hacked answer while the monitor reports late commitment is
                            consistent with answer-inert padding
  causal trace ablation     delete an interior span and force again; an unchanged answer
                            is evidence that the span was causally inert
  trace-length trajectory   whether hacking trajectories lengthen under the penalty
  evasion taxonomy          published obfuscation categories plus effort faking
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from config import ensure_output_dir, load_experiment_config
from env.dataset import load_problems
from env.generator import FakeGenerator
from eval.effort_faking import run_effort_faking_diagnostics
from eval.taxonomy import summarize_evasion_taxonomy
from reward.task_reward import compute_task_rewards


def _load_rollouts(path: Path) -> list[dict]:
    rows = []
    with path.open() as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _rollout_files(checkpoint_dir: Path) -> list[Path]:
    """The checkpoint's own rollouts, or every seed's rollouts for a replicated cell."""
    direct = checkpoint_dir / "rollouts.jsonl"
    if direct.exists():
        return [direct]
    return sorted(checkpoint_dir.glob("seed_*/rollouts.jsonl"))


def _find_policy_lora(checkpoint_dir: Path) -> Path:
    """Accept Atlas final/lora, trainer final/, step LoRAs, or a run root under default/."""
    root = checkpoint_dir.resolve()
    candidates = [
        root,
        root / "lora",
        root / "checkpoints" / "final" / "lora",
        root / "checkpoints" / "final",
        root / "final" / "lora",
        root / "final",
    ]
    default = root / "default"
    runs: list[Path] = []
    if default.is_dir():
        runs = sorted(
            default.glob("trigger_none_model_*_layers_*"),
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )
        for run in runs:
            candidates.extend(
                [
                    run / "checkpoints" / "final" / "lora",
                    run / "checkpoints" / "final",
                ]
            )
    for cand in candidates:
        if (cand / "adapter_config.json").is_file():
            return cand
    # Fallback: latest per-step LoRA (jobs that die before writing checkpoints/final).
    search_roots = [root / "checkpoints" / "lora", root / "lora"]
    for run in runs:
        search_roots.append(run / "checkpoints" / "lora")
    step_dirs: list[Path] = []
    for base in search_roots:
        if base.is_dir():
            step_dirs.extend(base.glob("step_*"))
    step_dirs = sorted(
        step_dirs,
        key=lambda p: int(p.name.split("_")[-1]) if p.name.split("_")[-1].isdigit() else -1,
        reverse=True,
    )
    for cand in step_dirs:
        if (cand / "adapter_config.json").is_file():
            return cand
    raise FileNotFoundError(f"No adapter_config.json under {root}")


def _resolve_lora(cfg) -> Path:
    """Priority: explicit lora_path → checkpoint_dir/metrics.json → heuristic search."""
    explicit = cfg.get("lora_path")
    if explicit:
        path = Path(str(explicit))
        if (path / "adapter_config.json").is_file():
            return path.resolve()
        raise FileNotFoundError(f"lora_path override has no adapter_config.json: {path}")

    ckpt = Path(str(cfg.get("checkpoint_dir") or ""))
    if not str(ckpt):
        raise ValueError(
            "E4 requires checkpoint_dir (e.g. results/seed_0/e2_scl)"
        )
    metrics_path = ckpt / "metrics.json"
    if metrics_path.exists():
        try:
            meta = json.loads(metrics_path.read_text())
        except json.JSONDecodeError:
            meta = {}
        recorded = meta.get("lora_path")
        if recorded:
            path = Path(str(recorded))
            if (path / "adapter_config.json").is_file():
                return path.resolve()
            print(f"E4: metrics.json lora_path missing adapter; falling back ({path})", flush=True)

    return _find_policy_lora(ckpt)


def _build_generator(cfg) -> object:
    if bool(cfg.get("dry_run", True)):
        return FakeGenerator(
            hack_rate=0.85,
            early_commit=True,
            seed=int(cfg.get("seed", 0)),
            env_variant=str(cfg.get("env_variant", "hardcode_visible")),
            policy_profile=str(cfg.get("policy_profile", "llama3-8b")),
        )
    from env.vllm_generator import VLLMHttpGenerator

    lora = _resolve_lora(cfg)
    print(f"E4 using LoRA={lora}", flush=True)
    return VLLMHttpGenerator(
        model_name=str(cfg.get("model_name", "meta-llama/Llama-3.1-8B-Instruct")),
        lora_path=lora,
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run E4 effort-faking diagnostics")
    parser.add_argument("overrides", nargs="*", help="overrides key=value")
    args = parser.parse_args(argv)

    cfg = load_experiment_config(os.environ.get("EXPERIMENT_NAME", "e4"), args.overrides)
    out = ensure_output_dir(cfg)

    if "checkpoint_dir" not in cfg or cfg.get("checkpoint_dir") in (None, ""):
        raise ValueError(
            "E4 requires checkpoint_dir (e.g. results/seed_0/e2_scl)"
        )
    ckpt = Path(str(cfg.get("checkpoint_dir")))
    env_variant = str(cfg.get("env_variant", "hardcode_visible"))
    gen = _build_generator(cfg)

    sources = _rollout_files(ckpt)
    per_source: list[dict] = []
    if sources:
        for path in sources:
            rows = _load_rollouts(path)[: int(cfg.get("num_prompts", 32))]
            per_source.append(
                {
                    "source": str(path),
                    **_diagnose(
                        cfg,
                        gen,
                        [r["prompt"] for r in rows],
                        [r["response"] for r in rows],
                        [bool(r.get("deceptive", False)) for r in rows],
                    ),
                }
            )
    else:
        problems = load_problems(
            offline=True, limit=int(cfg.get("num_prompts", 16)), env_variant=env_variant
        )
        prompts = [p.prompt for p in problems]
        responses = gen.generate(prompts)
        task = compute_task_rewards(
            responses,
            [p.test_list for p in problems],
            test_setup_codes=[p.test_setup_code for p in problems],
        )
        per_source.append(
            {"source": "fixtures", **_diagnose(cfg, gen, prompts, responses, task.deceptive)}
        )

    metrics = {
        "experiment": "e4",
        "checkpoint_dir": str(ckpt),
        "backend": "fake_generator" if bool(cfg.get("dry_run", True)) else "vllm_lora",
        "per_source": per_source,
        "n_sources": len(per_source),
    }
    with (out / "metrics.json").open("w") as f:
        json.dump(metrics, f, indent=2)

    print(f"E4 done → {out} ({len(per_source)} source(s))")
    for row in per_source:
        diag = row["diagnostics"]
        print(
            f"  {row['source']}: prefix_agreement={diag['prefix_agreement_rate']:.3f} "
            f"causal_inert={diag['causal_inert_rate']:.3f} "
            f"trace_len_hacking={diag.get('trace_len_mean_hacking')}"
        )
        print(f"    evasion={row['evasion_taxonomy']['counts']}")
    return 0


def _diagnose(cfg, generator, prompts: list[str], responses: list[str], labels: list[bool]) -> dict:
    prefix_fraction = float(cfg.get("prefix_fraction", 0.3))
    middle_fraction = float(cfg.get("ablation_middle_fraction", 0.4))
    match_rule = str(cfg.get("match_rule", "exact"))

    diagnostics = run_effort_faking_diagnostics(
        prompts,
        responses,
        generator,
        prefix_fraction=prefix_fraction,
        middle_fraction=middle_fraction,
        match_rule=match_rule,
        labels_deceptive=labels,
    )
    # Reuse per-row flags from the single diagnostic pass (no second GPU sweep).
    agreements = list(diagnostics.pop("prefix_agreements", []))
    inert = list(diagnostics.pop("causal_inertness", []))
    taxonomy = summarize_evasion_taxonomy(
        responses, prefix_agreements=agreements, causal_inertness=inert
    )
    taxonomy.pop("labels", None)
    return {"diagnostics": diagnostics, "evasion_taxonomy": taxonomy, "n": len(responses)}


if __name__ == "__main__":
    raise SystemExit(main())
