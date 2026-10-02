#!/usr/bin/env python3
"""E2 — Core monitor comparison (phase1 proposal §7).

One cell per invocation: M1 judge, M2 deception probe, M3 SCL, or M4 commitment probe,
at the low-KL trace-free condition. Phase 1 uses one matched pilot run per cell.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from config import ensure_output_dir, load_experiment_config
from training.atlas_grpo import run_atlas_grpo
from training.cell import print_cell_summary, run_cell


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run one E2 monitor cell")
    parser.add_argument("overrides", nargs="*", help="overrides e.g. monitor=scl beta=1e-4")
    args = parser.parse_args(argv)

    cfg = load_experiment_config(os.environ.get("EXPERIMENT_NAME", "e2"), args.overrides)
    out = ensure_output_dir(cfg)

    if not bool(cfg.get("dry_run", True)):
        summary = run_atlas_grpo(cfg, out, experiment="e2")
        print(
            f"E2 atlas GRPO → {out} monitor={summary['monitor']} "
            f"deception={summary.get('final_deception_rate')}"
        )
        return 0

    summary = run_cell(cfg, out, experiment="e2")
    print_cell_summary("e2", out, summary)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
