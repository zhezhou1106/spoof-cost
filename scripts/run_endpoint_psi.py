#!/usr/bin/env python3
"""Recompute the exact trained prefix score at the step-120 adapter.

At the final step the pre-update policy is the saved endpoint LoRA, so scoring
saved rollouts with that adapter reproduces the quantity the penalty would have
used. Calls the frozen SCL primitive unmodified: grid [0.1, 0.25, 0.5, 0.75, 0.9],
N=4, 64 tokens, whitespace-normalized equality, ψ=1 on unusable code, no b0.
Cut's trained readout is c_0.25 from the same curve.
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from config import ensure_output_dir, load_experiment_config
from env.generator import FakeGenerator
from monitors.factory import build_monitor


def _load_rollouts(path: Path) -> list[dict]:
    rows = []
    with path.open() as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _rollout_files(checkpoint_dir: Path) -> list[Path]:
    direct = checkpoint_dir / "rollouts.jsonl"
    if direct.exists():
        return [direct]
    return sorted(checkpoint_dir.glob("seed_*/rollouts.jsonl"))


def _find_policy_lora(checkpoint_dir: Path) -> Path:
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
    explicit = cfg.get("lora_path")
    if explicit:
        path = Path(str(explicit))
        if (path / "adapter_config.json").is_file():
            return path.resolve()
        raise FileNotFoundError(f"lora_path override has no adapter_config.json: {path}")

    ckpt = Path(str(cfg.get("checkpoint_dir") or ""))
    if not str(ckpt):
        raise ValueError("endpoint_psi requires checkpoint_dir")
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
            print(f"endpoint_psi: metrics.json lora_path missing adapter; falling back ({path})", flush=True)
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
    print(f"endpoint_psi using LoRA={lora}", flush=True)
    return VLLMHttpGenerator(
        model_name=str(cfg.get("model_name", "meta-llama/Llama-3.1-8B-Instruct")),
        lora_path=lora,
    )


def _summarize(values: list[float]) -> dict[str, float | None]:
    if not values:
        return {"median": None, "min": None, "max": None, "mean": None, "n": 0}
    return {
        "median": float(statistics.median(values)),
        "min": float(min(values)),
        "max": float(max(values)),
        "mean": float(statistics.mean(values)),
        "n": len(values),
    }


def _curve_at(curve: dict, fraction: float) -> float | None:
    if not curve:
        return None
    for key, val in curve.items():
        if abs(float(key) - fraction) < 1e-9:
            return float(val)
    return None


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Exact endpoint SCL/Cut readout")
    parser.add_argument("overrides", nargs="*", help="overrides key=value")
    args = parser.parse_args(argv)

    cfg = load_experiment_config(os.environ.get("EXPERIMENT_NAME", "track_a/endpoint_psi"), args.overrides)
    out = ensure_output_dir(cfg)

    ckpt = Path(str(cfg.get("checkpoint_dir") or ""))
    if not str(ckpt):
        raise ValueError("endpoint_psi requires checkpoint_dir")
    sources = _rollout_files(ckpt)
    if not sources:
        raise FileNotFoundError(f"No rollouts.jsonl under {ckpt}")

    gen = _build_generator(cfg)
    fractions = [float(x) for x in (cfg.get("fractions") or [0.1, 0.25, 0.5, 0.75, 0.9])]
    mon = build_monitor(
        "scl",
        generator=gen,
        match_rule=str(cfg.get("match_rule", "exact")),
        score_form=str(cfg.get("score_form", "auc")),
        fractions=fractions,
        num_continuations=int(cfg.get("num_continuations", 4)),
        forced_answer_tokens=int(cfg.get("forced_answer_tokens", 64)),
        baseline=None,
    )

    rows_out: list[dict] = []
    n_limit = int(cfg.get("num_prompts", 128))
    for path in sources:
        loaded = _load_rollouts(path)[:n_limit]
        print(f"endpoint_psi scoring {len(loaded)} rollouts from {path}", flush=True)
        for i, row in enumerate(loaded):
            score = mon.score(row["prompt"], row["response"])
            curve = score.extras.get("curve") or {}
            curve_f = {str(float(k)): float(v) for k, v in curve.items()}
            rec = {
                "source": str(path),
                "index": i,
                "psi": float(score.psi),
                "c_0.25": _curve_at(curve, 0.25),
                "curve": curve_f,
                "deceptive": bool(row.get("deceptive", False)),
                "passes_visible": row.get("passes_visible"),
                "passes_all": row.get("passes_all"),
            }
            rows_out.append(rec)
            if (i + 1) % 16 == 0:
                print(f"  {i + 1}/{len(loaded)} psi={score.psi:.3f} c025={rec['c_0.25']}", flush=True)

    psis = [float(r["psi"]) for r in rows_out]
    cuts = [float(r["c_0.25"]) for r in rows_out if r["c_0.25"] is not None]
    metrics = {
        "experiment": "endpoint_psi",
        "checkpoint_dir": str(ckpt.resolve()),
        "backend": "fake_generator" if bool(cfg.get("dry_run", True)) else "vllm_lora",
        "fractions": fractions,
        "num_continuations": int(cfg.get("num_continuations", 4)),
        "forced_answer_tokens": int(cfg.get("forced_answer_tokens", 64)),
        "baseline": None,
        "n": len(rows_out),
        "psi": _summarize(psis),
        "c_0.25": _summarize(cuts),
        "note": (
            "Unmodified SCL primitive on endpoint rollouts + endpoint adapter. "
            "Cut's trained score is c_0.25; SCL's trained score is psi (AUC of the curve)."
        ),
    }
    jsonl_path = out / "endpoint_psi.jsonl"
    with jsonl_path.open("w") as f:
        for rec in rows_out:
            f.write(json.dumps(rec) + "\n")
    with (out / "metrics.json").open("w") as f:
        json.dump(metrics, f, indent=2)

    print(
        f"endpoint_psi done → {out} n={len(rows_out)} "
        f"psi median={metrics['psi']['median']} [{metrics['psi']['min']}, {metrics['psi']['max']}] "
        f"c_0.25 median={metrics['c_0.25']['median']}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
