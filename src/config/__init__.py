"""Shared config helpers."""

from __future__ import annotations

from pathlib import Path

from omegaconf import DictConfig, OmegaConf

REPO_ROOT = Path(__file__).resolve().parents[2]
CONFIG_DIR = REPO_ROOT / "configs"


def load_experiment_config(name: str, overrides: list[str] | None = None) -> DictConfig:
    """Load configs/experiments/{name}.yaml with optional dotlist overrides.

    Layers, later winning: pre-registered constants, then the CPU dry-run profile,
    then the experiment file, then the GPU profile when the run is not a dry run, then
    command-line overrides. The dry-run profile only downgrades cost knobs (seeds,
    steps, prompt count, reference-baseline estimation); `gpu.yaml` restores each of
    them, so the two profiles cannot silently drift apart.
    """
    path = CONFIG_DIR / "experiments" / f"{name}.yaml"
    cfg = OmegaConf.load(path)
    # YAML may include unused Hydra-style defaults; drop them for plain OmegaConf merge.
    if "defaults" in cfg:
        del cfg["defaults"]
    pre = OmegaConf.load(CONFIG_DIR / "preregistered.yaml")
    dry = OmegaConf.load(CONFIG_DIR / "dry_run.yaml")
    override_cfg = OmegaConf.from_dotlist(overrides) if overrides else OmegaConf.create({})

    merged = OmegaConf.merge(pre, dry, cfg)
    requested_dry_run = override_cfg.get("dry_run", merged.get("dry_run", True))
    if not bool(requested_dry_run):
        merged = OmegaConf.merge(merged, OmegaConf.load(CONFIG_DIR / "gpu.yaml"))
    return OmegaConf.merge(merged, override_cfg)


def ensure_output_dir(cfg: DictConfig) -> Path:
    out = Path(str(cfg.get("output_dir", "results/out")))
    out.mkdir(parents=True, exist_ok=True)
    OmegaConf.save(cfg, out / "config.yaml")
    return out
