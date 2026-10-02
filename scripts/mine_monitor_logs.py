#!/usr/bin/env python3
"""Scan on-disk E2 logs for raw ψ vs applied α·max(z,0)."""

from __future__ import annotations

import json
import statistics
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

_R = ROOT / "results"
CELLS = [
    ("none", None, _R / "e0"),
    ("probe", 0, _R / "seed_0" / "e2_probe"),
    ("probe", 1, _R / "seed_1" / "e2_probe"),
    ("probe", 2, _R / "seed_2" / "e2_probe"),
    ("scl", 0, _R / "seed_0" / "e2_scl"),
    ("scl", 1, _R / "seed_1" / "e2_scl"),
    ("scl", 2, _R / "seed_2" / "e2_scl"),
    ("cprobe", 0, _R / "seed_0" / "e2_cprobe"),
    ("cprobe", 1, _R / "seed_1" / "e2_cprobe"),
    ("cprobe", 2, _R / "seed_2" / "e2_cprobe"),
]

PSI_KEYS = ("psi", "monitor_psi", "psi_mean", "psi_median", "rewards_metrics/monitor_psi")


def _load_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    rows = []
    with path.open() as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _has_raw_psi(row: dict) -> bool:
    for key in PSI_KEYS:
        if key in row and key != "monitor_term_mean":
            val = row[key]
            if isinstance(val, (int, float)) or (isinstance(val, list) and val):
                return True
    return False


def _scan_logs(patterns: list[str]) -> dict[str, int]:
    logs = ROOT / "logs"
    hits = {p: 0 for p in patterns}
    if not logs.is_dir():
        return hits
    for path in logs.glob("*.out"):
        try:
            text = path.read_text(errors="replace")
        except OSError:
            continue
        for p in patterns:
            if p in text:
                hits[p] += 1
    return hits


def main() -> int:
    print("# Monitor-log mine\n")
    print("| Cell | n_steps | term_mean mean [min,max] | raw ψ on disk |")
    print("|---|---:|---|---|")
    any_raw = False
    for arm, seed, cell in CELLS:
        press = _load_jsonl(cell / "pressure_trajectory.jsonl")
        label = f"{arm}" if seed is None else f"{arm} s{seed}"
        if not press:
            print(f"| {label} | 0 | — | no pressure file |")
            continue
        terms = [float(r["monitor_term_mean"]) for r in press if r.get("monitor_term_mean") is not None]
        raw = any(_has_raw_psi(r) for r in press)
        any_raw = any_raw or raw
        if terms:
            summary = f"{statistics.mean(terms):.3f} [{min(terms):.3f}, {max(terms):.3f}]"
        else:
            summary = "—"
        print(f"| {label} | {len(press)} | {summary} | {'yes' if raw else 'no'} |")

        metrics = cell / "metrics.json"
        if metrics.exists():
            meta = json.loads(metrics.read_text())
            pressure = meta.get("pressure") or {}
            if "monitor_term_mean_mean" in pressure:
                print(
                    f"  metrics.pressure.monitor_term_mean_mean="
                    f"{pressure['monitor_term_mean_mean']:.3f}"
                )

    log_hits = _scan_logs(["monitor_psi", "\"psi\":", "psi_mean", "rewards_metrics/monitor_psi"])
    print("\nLog substring hits:", log_hits)

    wandb = list((ROOT / "artifacts").glob("**/wandb/**/*")) if (ROOT / "artifacts").exists() else []
    print(f"Local W&B artifact files found: {len(wandb)}")

    verdict = (
        "RAW_PSI"
        if any_raw
        else "PENALTY_ONLY"
    )
    print(f"\nVerdict: {verdict}")
    if verdict == "PENALTY_ONLY":
        print(
            "Only α·max(z,0) was retained. Within-group z-scoring makes that "
            "term scale-free; it does not recover the level of ψ."
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
