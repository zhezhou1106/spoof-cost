#!/usr/bin/env python3
"""Plot trained prefix readout (ψ or c_0.25) vs endpoint hack rate.

Prefers endpoint_psi/metrics.json. Falls back to E4 A0.30 if the exact
score is not on disk yet.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

def _cells() -> list[tuple[str, str, int, Path, Path, Path]]:
    results = ROOT / "results"
    rows: list[tuple[str, str, int, Path, Path, Path]] = [
        (
            "none",
            "No monitor",
            0,
            results / "e0",
            results / "e4_e0",
            results / "endpoint_psi_e0",
        )
    ]
    for arm, label in (("probe", "Probe"), ("scl", "SCL"), ("cprobe", "Cut")):
        for seed in (0, 1, 2):
            seed_root = results / f"seed_{seed}"
            rows.append(
                (
                    arm,
                    label,
                    seed,
                    seed_root / f"e2_{arm}",
                    seed_root / f"e4_{arm}",
                    seed_root / f"e2_{arm}" / "endpoint_psi",
                )
            )
    return rows


CELLS = _cells()

COLOR = {"none": "#4C78A8", "probe": "#F58518", "scl": "#54A24B", "cprobe": "#E45756"}
MARKER = {0: "o", 1: "s", 2: "D"}


def _hack(cell: Path) -> float:
    meta = json.loads((cell / "metrics.json").read_text())
    return float(meta.get("final_deception_rate", meta.get("hack_rate", 0.0)))


def _e4_prefix(e4: Path) -> float | None:
    p = e4 / "metrics.json"
    if not p.exists():
        return None
    meta = json.loads(p.read_text())
    src = (meta.get("per_source") or [{}])[0]
    return float(src.get("diagnostics", {}).get("prefix_agreement_rate"))


def _exact(arm: str, epsi: Path) -> tuple[float | None, str]:
    p = epsi / "metrics.json"
    if not p.exists():
        return None, "missing"
    meta = json.loads(p.read_text())
    if arm == "cprobe":
        block = meta.get("c_0.25") or {}
        key = "c_0.25"
    else:
        block = meta.get("psi") or {}
        key = "psi"
    val = block.get("median")
    return (float(val) if val is not None else None), key


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=ROOT / "results" / "aggregate" / "prefix_vs_hack.png")
    args = parser.parse_args(argv)

    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("[plot] matplotlib missing", flush=True)
        return 1

    fig, ax = plt.subplots(figsize=(6.4, 4.6))
    used_exact = 0
    print("| run | d | exact | source | A0.30 |")
    print("|---|---:|---:|---|---:|")
    # Draw high-d points first so mixed-run markers stay visible on the left stack.
    rows = []
    for arm, label, seed, cell, e4, epsi in CELLS:
        d = _hack(cell)
        exact, src = _exact(arm, epsi)
        a030 = _e4_prefix(e4)
        y, ysrc = (exact, src) if exact is not None else (a030, "A0.30")
        if exact is not None:
            used_exact += 1
        name = label if arm == "none" else f"{label} s{seed}"
        print(f"| {name} | {d:.3f} | {'' if exact is None else f'{exact:.3f}'} | {ysrc} | {'' if a030 is None else f'{a030:.3f}'} |")
        rows.append((d, arm, seed, name, y))
    for d, arm, seed, name, y in sorted(rows, key=lambda r: -r[0]):
        if y is None:
            continue
        ax.scatter(
            y,
            d,
            color=COLOR[arm],
            marker=MARKER.get(seed, "o") if arm != "none" else "X",
            s=92,
            zorder=3,
            label=name,
            edgecolors="white",
            linewidths=0.7,
        )
    ax.set_xlabel(
        r"Endpoint readout (SCL $\psi$ / Cut $c_{0.25}$)"
        if used_exact
        else r"Prefix agreement $A_{0.30}$ (fallback)"
    )
    ax.set_ylabel(r"Hack rate $d$")
    ax.set_xlim(-0.06, 1.06)
    ax.set_ylim(-0.04, 1.06)
    ax.legend(
        loc="lower right",
        frameon=True,
        fontsize=7.5,
        handletextpad=0.4,
        borderpad=0.5,
        labelspacing=0.35,
        title="Run",
        title_fontsize=8,
    )
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"[plot] wrote {args.out} exact={used_exact}/10", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
