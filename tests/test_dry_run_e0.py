"""Dry-run integration: E0 produces metrics with required keys."""

import json
from pathlib import Path

from config import load_experiment_config
from training import DryRunTrainer


def test_load_all_experiment_configs():
    for name in ["track_a/e0", "track_a/e1", "track_a/e2", "track_a/e4", "track_a/e5"]:
        cfg = load_experiment_config(name)
        exp = str(cfg.get("experiment", name))
        assert exp in {name.split("/")[-1], name.replace("_", "")} or name.split("/")[-1] in exp


def test_track_a_e0_config_trace_free():
    cfg = load_experiment_config("track_a/e0")
    assert str(cfg.get("output_dir")) == "results/e0"
    assert bool(cfg.get("resume_from_checkpoint", True)) is False
    assert bool(cfg.get("cot_min_reasoning_in_gate")) is True
    assert bool(cfg.get("cot_require_problem_tokens")) is False
    assert int(cfg.get("cot_min_reasoning_chars", 0)) == 200
    assert int(cfg.get("cot_min_reasoning_words", 4)) == 40


def test_dry_run_e0(tmp_path: Path):
    trainer = DryRunTrainer(
        monitor_name="none",
        group_size=4,
        num_steps=1,
        num_prompts=4,
        hack_rate=0.95,
        seed=0,
    )
    result = trainer.run(tmp_path / "e0")
    assert (tmp_path / "e0" / "metrics.json").exists()
    assert (tmp_path / "e0" / "rollouts.jsonl").exists()
    m = result.metrics
    for key in [
        "final_deception_rate",
        "gate_deception_pass",
        "steps",
        "n_rollouts",
    ]:
        assert key in m
    assert m["n_rollouts"] == 4 * 4  # prompts * group_size


def test_dry_run_with_scl(tmp_path: Path):
    trainer = DryRunTrainer(
        monitor_name="scl",
        group_size=2,
        num_steps=1,
        num_prompts=2,
        alpha=3.0,
        hack_rate=0.9,
        seed=1,
    )
    result = trainer.run(tmp_path / "scl")
    assert result.metrics["monitor"] == "scl"
    data = json.loads((tmp_path / "scl" / "metrics.json").read_text())
    assert data["steps"][0]["psi_mean"] >= 0.0
