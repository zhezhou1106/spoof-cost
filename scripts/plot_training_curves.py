#!/usr/bin/env python3
"""Plot multi-seed training curves from outcome_trajectory.jsonl files.

Writes ``results/aggregate/training_curves.png``.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

ARM_STYLE = {
    "none": ("No monitor", "e0", "#4C78A8"),
    "probe": ("Probe", "e2_probe", "#F58518"),
    "scl": ("SCL", "e2_scl", "#54A24B"),
    "cprobe": ("Cut", "e2_cprobe", "#E45756"),
}


def _smooth(values: list[float], window: int) -> list[float]:
    """Gaussian smooth along steps. ``window=1`` is a no-op; ``sigma = window/4``."""
    import numpy as np
    from scipy.ndimage import gaussian_filter1d

    y = np.asarray(values, dtype=float)
    if window <= 1 or y.size == 0:
        return y.tolist()
    sigma = max(float(window) / 4.0, 0.5)
    return gaussian_filter1d(y, sigma=sigma, mode="nearest").tolist()


def _load_traj(path: Path) -> list[dict]:
    if not path.exists():
        return []
    rows = []
    with path.open() as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return sorted(rows, key=lambda r: int(r.get("step", 0)))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Plot multi-seed training curves")
    parser.add_argument(
        "--track-root",
        type=Path,
        default=ROOT / "results",
    )
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument(
        "--out",
        type=Path,
        default=ROOT / "results" / "aggregate" / "training_curves.png",
    )
    parser.add_argument(
        "--smooth-window",
        type=int,
        default=11,
        help="Gaussian window; sigma = window/4. Set 1 to disable smoothing.",
    )
    args = parser.parse_args(argv)

    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("[plot] matplotlib not installed; skipping", flush=True)
        return 1

    fig, axes = plt.subplots(1, 2, figsize=(10, 3.6), sharex=True)
    ax_v, ax_h = axes
    any_data = False

    for arm, (label, dirname, color) in ARM_STYLE.items():
        seeds = [None] if arm == "none" else args.seeds
        for seed in seeds:
            if arm == "none":
                traj = _load_traj(args.track_root / "e0" / "outcome_trajectory.jsonl")
                ls = "-"
                legend = f"{label} (E0)"
            else:
                traj = _load_traj(
                    args.track_root / f"seed_{seed}" / dirname / "outcome_trajectory.jsonl"
                )
                ls = ["-", "--", ":"][int(seed) % 3]
                legend = f"{label} s{seed}"
            if not traj:
                continue
            steps = [int(r["step"]) for r in traj if r.get("visible_pass_rate") is not None]
            vis = [float(r["visible_pass_rate"]) for r in traj if r.get("visible_pass_rate") is not None]
            held = [float(r["heldout_pass_rate"]) for r in traj if r.get("heldout_pass_rate") is not None]
            steps_h = [int(r["step"]) for r in traj if r.get("heldout_pass_rate") is not None]
            if vis:
                ax_v.plot(
                    steps,
                    _smooth(vis, args.smooth_window),
                    color=color,
                    linestyle=ls,
                    alpha=0.95,
                    lw=2.0,
                    label=legend,
                )
                any_data = True
            if held:
                ax_h.plot(
                    steps_h,
                    _smooth(held, args.smooth_window),
                    color=color,
                    linestyle=ls,
                    alpha=0.95,
                    lw=2.0,
                    label=legend,
                )
                any_data = True

    ax_v.set_title("Visible-pass rate")
    ax_h.set_title("Held-out-pass rate")
    ax_v.set_xlabel("GRPO step")
    ax_h.set_xlabel("GRPO step")
    ax_v.set_ylabel("Rate")
    ax_v.set_ylim(-0.02, 1.02)
    ax_h.set_ylim(-0.02, 1.02)
    ax_v.axhline(0.10, color="gray", lw=0.6, ls=":")
    ax_h.axhline(0.10, color="gray", lw=0.6, ls=":")
    handles, labels = ax_v.get_legend_handles_labels()
    # Deduplicate legend labels while preserving order.
    seen = set()
    uniq = [(h, l) for h, l in zip(handles, labels, strict=True) if not (l in seen or seen.add(l))]
    if uniq:
        fig.legend(
            [h for h, _ in uniq],
            [l for _, l in uniq],
            loc="upper center",
            ncol=4,
            fontsize=7,
            frameon=False,
            bbox_to_anchor=(0.5, 1.08),
        )
    fig.tight_layout()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    if not any_data:
        # Placeholder so the paper path exists before GPU runs finish.
        ax_v.text(0.5, 0.5, "No outcome_trajectory.jsonl yet", ha="center", va="center", transform=ax_v.transAxes)
        ax_h.text(0.5, 0.5, "Run E0 then E2 seeds first", ha="center", va="center", transform=ax_h.transAxes)
    fig.savefig(args.out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"[plot] wrote {args.out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
