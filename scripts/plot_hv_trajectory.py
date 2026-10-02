#!/usr/bin/env python3
"""Plot h/v over training for mixed runs, plus a printed endpoint h/v table."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

_R = ROOT / "results"
ENDPOINT = [
    ("No monitor", None, _R / "e0"),
    ("Probe s0", "hacking", _R / "seed_0" / "e2_probe"),
    ("Probe s1", "hacking", _R / "seed_1" / "e2_probe"),
    ("Probe s2", "hacking", _R / "seed_2" / "e2_probe"),
    ("SCL s0", "mixed", _R / "seed_0" / "e2_scl"),
    ("SCL s1", "hacking", _R / "seed_1" / "e2_scl"),
    ("SCL s2", "hacking", _R / "seed_2" / "e2_scl"),
    ("Cut s0", "hacking", _R / "seed_0" / "e2_cprobe"),
    ("Cut s1", "mixed", _R / "seed_1" / "e2_cprobe"),
    ("Cut s2", "mixed", _R / "seed_2" / "e2_cprobe"),
]

MIXED = [
    ("SCL s0", "#54A24B", "-", _R / "seed_0" / "e2_scl"),
    ("Cut s1", "#E45756", "--", _R / "seed_1" / "e2_cprobe"),
    ("Cut s2", "#E45756", ":", _R / "seed_2" / "e2_cprobe"),
]


def _load_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    rows = []
    with path.open() as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return sorted(rows, key=lambda r: int(r.get("step", 0)))


def _endpoint_hv(cell: Path) -> tuple[float | None, float | None, float | None]:
    metrics = cell / "metrics.json"
    if metrics.exists():
        meta = json.loads(metrics.read_text())
        h = meta.get("all_pass_rate")
        v = meta.get("visible_pass_rate")
        if h is None:
            h = meta.get("final_heldout_pass_rate")
        if h is not None and v is not None:
            h_f, v_f = float(h), float(v)
            return h_f, v_f, (h_f / v_f if v_f > 0 else None)
    # Fallback: last outcome_trajectory row, or rollout counts.
    traj = _load_jsonl(cell / "outcome_trajectory.jsonl")
    if traj:
        last = traj[-1]
        h = last.get("heldout_pass_rate")
        v = last.get("visible_pass_rate")
        if h is not None and v is not None:
            h_f, v_f = float(h), float(v)
            return h_f, v_f, (h_f / v_f if v_f > 0 else None)
    return None, None, None


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="h/v table and mixed-run trajectory")
    parser.add_argument(
        "--out",
        type=Path,
        default=ROOT / "results" / "aggregate" / "hv_trajectory.png",
    )
    parser.add_argument("--early-lo", type=int, default=10)
    parser.add_argument("--early-hi", type=int, default=27)
    args = parser.parse_args(argv)

    print("| Run | h | v | h/v |")
    print("|---|---:|---:|---:|")
    table = []
    for name, _regime, cell in ENDPOINT:
        h, v, hv = _endpoint_hv(cell)
        table.append((name, h, v, hv))
        def fmt(x):
            return "—" if x is None else f"{x:.3f}"
        print(f"| {name} | {fmt(h)} | {fmt(v)} | {fmt(hv)} |")

    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("[plot] matplotlib not installed; skipping figure", flush=True)
        return 0

    fig, ax = plt.subplots(figsize=(6.4, 3.4))
    early_vals: list[float] = []
    for label, color, ls, cell in MIXED:
        traj = _load_jsonl(cell / "outcome_trajectory.jsonl")
        steps, hvs, vs = [], [], []
        for row in traj:
            v = row.get("visible_pass_rate")
            h = row.get("heldout_pass_rate")
            step = int(row.get("step", 0))
            if v is None or h is None or float(v) <= 0:
                continue
            ratio = float(h) / float(v)
            steps.append(step)
            hvs.append(ratio)
            vs.append(float(v))
            if args.early_lo <= step <= args.early_hi:
                early_vals.append(ratio)
        if steps:
            ax.plot(steps, hvs, color=color, linestyle=ls, lw=2.0, label=label)
    if early_vals:
        lo, hi = min(early_vals), max(early_vals)
        ax.axvspan(args.early_lo, args.early_hi, color="#bbbbbb", alpha=0.25, label="steps 10–27")
        ax.axhline(sum(early_vals) / len(early_vals), color="#888888", lw=0.8, ls="--")
        print(
            f"\nEarly-band h/v (steps {args.early_lo}–{args.early_hi}, mixed runs): "
            f"mean={sum(early_vals)/len(early_vals):.3f} [{lo:.3f}, {hi:.3f}]"
        )
    ax.set_xlabel("GRPO step")
    ax.set_ylabel(r"$h/v$")
    ax.set_ylim(-0.02, 1.02)
    ax.set_xlim(0, 120)
    ax.legend(fontsize=8, frameon=False)
    fig.tight_layout()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"[plot] wrote {args.out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
