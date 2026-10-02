#!/usr/bin/env python3
"""Estimate π₀ SCL reference baseline b₀(x) for a finished E0 run (proposal §5)."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from config import ensure_output_dir, load_experiment_config  # noqa: E402
from env.vllm_generator import VLLMHttpGenerator  # noqa: E402

# Import after src is on path; run_e0 lives in scripts/.
sys.path.insert(0, str(ROOT / "scripts"))
from run_e0 import estimate_and_save_reference_baseline  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Write E0 reference_baseline.json under π₀")
    parser.add_argument("overrides", nargs="*", help="Hydra-style overrides key=value")
    args = parser.parse_args(argv)
    cfg = load_experiment_config(os.environ.get("EXPERIMENT_NAME", "e0"), args.overrides)
    out = ensure_output_dir(cfg)
    gen = VLLMHttpGenerator(
        model_name=str(cfg.get("model_name", "meta-llama/Llama-3.1-8B-Instruct")),
        lora_path=None,
    )
    info = estimate_and_save_reference_baseline(cfg, out, gen)
    metrics_path = out / "metrics.json"
    if metrics_path.exists():
        metrics = json.loads(metrics_path.read_text())
        metrics["reference_baseline"] = info
        with metrics_path.open("w") as f:
            json.dump(metrics, f, indent=2)
    print(json.dumps(info, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
