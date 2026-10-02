#!/usr/bin/env python3
"""Sample 30 capability-failure endpoint rollouts from each mixed run."""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from env.parse import parse_response
from eval.taxonomy import classify_rollout_outcome

MIXED = [
    ("SCL s0", ROOT / "results" / "seed_0" / "e2_scl" / "rollouts.jsonl"),
    ("Cut s1", ROOT / "results" / "seed_1" / "e2_cprobe" / "rollouts.jsonl"),
    ("Cut s2", ROOT / "results" / "seed_2" / "e2_cprobe" / "rollouts.jsonl"),
]


def _load(path: Path) -> list[dict]:
    rows = []
    with path.open() as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _first_lines(text: str, n: int = 8) -> str:
    lines = [ln.rstrip() for ln in (text or "").splitlines() if ln.strip()]
    return "\n".join(lines[:n])


def _prompt_id(prompt: str) -> str:
    # User turn is the MBPP problem sentence.
    if "Write a python function" in prompt:
        start = prompt.index("Write a python function")
        chunk = prompt[start:].split("\n", 1)[0]
        return chunk[:160]
    return prompt[:160].replace("\n", " ")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=30)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--out",
        type=Path,
        default=ROOT / "results" / "aggregate" / "capability_failure_review.md",
    )
    args = parser.parse_args(argv)

    rng = random.Random(args.seed)
    parts = [
        "# Capability-failure review (mixed runs)",
        "",
        "Draft labels for spot-check. Categories: `attempted_general_bug` / "
        "`truncated` / `off_task` / `degenerate`.",
        "",
        f"Sample size: {args.n} per run; RNG seed {args.seed}.",
        "",
    ]
    dump = []
    for run, path in MIXED:
        rows = _load(path)
        fails = [(i, r) for i, r in enumerate(rows) if classify_rollout_outcome(r) == "capability_failure"]
        sample = fails if len(fails) <= args.n else rng.sample(fails, args.n)
        sample = sorted(sample, key=lambda t: t[0])
        parts.append(f"## {run} ({len(fails)} capability_failure; showing {len(sample)})")
        parts.append("")
        for k, (idx, row) in enumerate(sample, start=1):
            parsed = parse_response(row.get("response", ""))
            reason = parsed.reasoning or ""
            code = parsed.code or ""
            rec = {
                "run": run,
                "index": idx,
                "k": k,
                "prompt": _prompt_id(row.get("prompt", "")),
                "reasoning": reason,
                "code": code,
                "format_ok": row.get("format_ok"),
                "reasoning_chars": row.get("reasoning_chars"),
            }
            dump.append(rec)
            parts.append(f"### {run} #{k} (rollout {idx})")
            parts.append("")
            parts.append(f"- **prompt:** {_prompt_id(row.get('prompt', ''))}")
            parts.append(f"- **format_ok:** {row.get('format_ok')}  **chars:** {row.get('reasoning_chars')}")
            parts.append(f"- **proposed_label:** TODO")
            parts.append("")
            parts.append("<reasoning>")
            parts.append(_first_lines(reason, 8) or "(empty)")
            parts.append("</reasoning>")
            parts.append("")
            parts.append("```python")
            parts.append((code or "(no code)")[:800])
            parts.append("```")
            parts.append("")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text("\n".join(parts) + "\n")
    json_path = args.out.with_suffix(".json")
    json_path.write_text(json.dumps(dump, indent=2))
    print(f"wrote {args.out} and {json_path} ({len(dump)} items)", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
