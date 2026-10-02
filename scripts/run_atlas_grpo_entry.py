"""Hydra entry for Spoof Cost → atlas GRPO.

Env (set by training.atlas_grpo.run_atlas_grpo):
  SPOOF_ENV_VARIANT, SPOOF_MONITOR, SPOOF_ALPHA, SPOOF_GROUP_SIZE, SPOOF_OUTPUT_DIR,
  SPOOF_EXPERIMENT, SPOOF_DECEPTION_GATE, SCL knobs, SPOOF_INDOMAIN_PROBE_PATH
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import hydra
from omegaconf import DictConfig, OmegaConf

from obfuscation_atlas.config import register_configs
from obfuscation_atlas.detectors import evaluate as atlas_evaluate
from obfuscation_atlas.scripts import train_grpo_model_against_probe as atlas_rl
from obfuscation_atlas.training import obfuscation_training as atlas_ot
from training.atlas_grpo import install_spoof_reward_and_eval_hooks, register_spoof_cot_preparer

register_configs()

_orig_get_layers_below_loss_threshold = atlas_evaluate.get_layers_below_loss_threshold
_orig_train_and_eval_multiple_detectors = atlas_evaluate.train_and_eval_multiple_detectors


def _safe_get_layers_below_loss_threshold(train_losses, loss_threshold):
    """Do not abort RL after a finished run if no probe layer beats Atlas's cutoff."""
    try:
        return _orig_get_layers_below_loss_threshold(train_losses, loss_threshold)
    except ValueError as exc:
        layers = []
        for key, losses in (train_losses or {}).items():
            if not key.startswith("layer_") or not key.endswith("_loss"):
                continue
            layer = int(key.replace("layer_", "").replace("_loss", ""))
            if losses:
                layers.append(layer)
        layers = sorted(set(layers))
        print(
            f"[spoof-cost] {exc}; using all {len(layers)} probe layers instead of aborting",
            flush=True,
        )
        if not layers:
            raise
        return layers


def _fake_layer_metrics(layer: int) -> dict[str, Any]:
    """Minimal metrics so Atlas threshold helpers and the startup violin log do not crash.

    In-domain injection never scored a holdout, so dummy ``scores`` keep
    ``log_detector_metrics`` / seaborn from seeing an empty Score column.
    """
    del layer
    return {
        "metrics_at_threshold": {"threshold": 0.5, "tpr": 0.8, "fpr": 0.01},
        "mean_score_positive": 0.8,
        "mean_score_negative": 0.2,
        "auroc": 0.75,
        "scores": {
            "positive": [0.85, 0.80, 0.78, 0.82],
            "negative": [0.20, 0.18, 0.25, 0.22],
        },
    }


def _inject_indomain_probe(**kwargs: Any) -> dict[str, Any]:
    """Skip off-domain Atlas pretrain; return E1 in-domain LinearProbe dict."""
    from eval.indomain_probe import load_as_atlas_linear_probes

    path = os.environ["SPOOF_INDOMAIN_PROBE_PATH"]
    probes = load_as_atlas_linear_probes(path)
    layers = kwargs.get("layers") or list(probes.keys())
    # Restrict to fitted layers only.
    layer_metrics = {str(li): _fake_layer_metrics(int(li)) for li in probes}
    train_dynamics = {f"layer_{li}_loss": [0.1] for li in probes}
    obfuscate_overs = kwargs.get("obfuscate_over") or ["last-token-generation"]
    if isinstance(obfuscate_overs, str):
        obfuscate_overs = [obfuscate_overs]
    results: dict[str, Any] = {}
    for obf in obfuscate_overs:
        key = f"linear-probe_{str(obf).replace('_', '-')}"
        # Third slot is train_eval_metrics. None (not {}) skips Atlas's train-score hist.
        results[key] = (probes, (layer_metrics, train_dynamics, None))
    print(
        f"[spoof-cost] Injecting in-domain probe from {path} "
        f"(layers={sorted(probes.keys())}; skipping off-domain pretrain)",
        flush=True,
    )
    del layers  # unused; Atlas uses activation_matching_layers from config
    return results


def _maybe_patched_train_and_eval(**kwargs: Any) -> dict[str, Any]:
    monitor = os.environ.get("SPOOF_MONITOR", "none").lower()
    path = os.environ.get("SPOOF_INDOMAIN_PROBE_PATH", "").strip()
    if path and monitor in {"probe", "m2", "deception"}:
        return _inject_indomain_probe(**kwargs)
    return _orig_train_and_eval_multiple_detectors(**kwargs)


atlas_evaluate.get_layers_below_loss_threshold = _safe_get_layers_below_loss_threshold
atlas_rl.get_layers_below_loss_threshold = _safe_get_layers_below_loss_threshold
atlas_ot.get_layers_below_loss_threshold = _safe_get_layers_below_loss_threshold
atlas_evaluate.train_and_eval_multiple_detectors = _maybe_patched_train_and_eval
atlas_rl.train_and_eval_multiple_detectors = _maybe_patched_train_and_eval


def _install_policy_lora_warmstart() -> None:
    """Optional: load policy LoRA weights from SPOOF_POLICY_WARMSTART_LORA after trainer init.

    Used when a full trainer checkpoint (optimizer/RNG) is missing but a LoRA adapter
    exists (e.g. Cut s0 resume from pre-collapse step_25). Reference adapter stays at
    the scratch/base init; only the trainable policy adapter is replaced.
    """
    warm = os.environ.get("SPOOF_POLICY_WARMSTART_LORA", "").strip()
    if not warm:
        return
    from obfuscation_atlas.training.grpo_trainer import GRPOObfuscationTrainer

    _orig_init = GRPOObfuscationTrainer.__init__

    def _init(self, *args, **kwargs):  # type: ignore[no-untyped-def]
        _orig_init(self, *args, **kwargs)
        path = Path(os.environ["SPOOF_POLICY_WARMSTART_LORA"]).resolve()
        weight_file = path / "adapter_model.bin"
        if not weight_file.is_file():
            raise FileNotFoundError(f"SPOOF_POLICY_WARMSTART_LORA missing adapter_model.bin: {path}")
        import torch
        from peft import set_peft_model_state_dict

        state = torch.load(weight_file, map_location="cpu", weights_only=True)
        model = self.state.model.shared_base_model
        incompatible = set_peft_model_state_dict(model, state, adapter_name="policy")
        print(
            f"[spoof-cost] Warm-started policy LoRA from {path} "
            f"(keys={len(state)}; incompatible={incompatible})",
            flush=True,
        )

    GRPOObfuscationTrainer.__init__ = _init  # type: ignore[method-assign]


@hydra.main(version_base=None, config_name="grpo_code_gen")
def main(cfg: DictConfig) -> None:
    env_variant = os.environ.get("SPOOF_ENV_VARIANT", "hardcode_visible")
    register_spoof_cot_preparer(env_variant)
    _install_policy_lora_warmstart()
    install_spoof_reward_and_eval_hooks(
        monitor=os.environ.get("SPOOF_MONITOR", "none"),
        alpha=float(os.environ.get("SPOOF_ALPHA", "0")),
        group_size=int(os.environ.get("SPOOF_GROUP_SIZE", "8")),
        output_dir=os.environ.get("SPOOF_OUTPUT_DIR", "results/atlas"),
        experiment=os.environ.get("SPOOF_EXPERIMENT", "atlas"),
        deception_gate=float(os.environ.get("SPOOF_DECEPTION_GATE", "0.80")),
    )
    cfg_obj = OmegaConf.to_object(cfg)
    atlas_rl.run_rl(cfg_obj)  # type: ignore[arg-type]


if __name__ == "__main__":
    main()
