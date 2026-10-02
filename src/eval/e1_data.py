"""Load and label E0 checkpoint rollouts for the E1 transfer gate."""

from __future__ import annotations

import json
import re
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from reward.task_reward import compute_task_rewards

_VISIBLE_RE = re.compile(r"Visible test case:\s*\n(assert[^\n]+)")
_USER_TEXT_RE = re.compile(
    r"<\|start_header_id\|>user<\|end_header_id\|>\s*\n\n(.*?)\n\nVisible test case:",
    re.DOTALL,
)


@dataclass
class LabeledRollout:
    prompt: str
    response: str
    deceptive: bool
    passes_visible: bool
    passes_all: bool
    step: int
    test_list: list[str]
    test_setup_code: str


def find_e0_run_dir(checkpoint_dir: str | Path) -> Path:
    """Prefer the newest atlas run under checkpoint_dir/default with responses + final LoRA.

    If live ``checkpoints/responses`` were pruned, restore from
    ``{checkpoint_dir}/responses_archive`` (written by scripts/run.sh after E0).
    """
    import shutil

    root = Path(checkpoint_dir)
    default = root / "default"
    if not default.is_dir():
        raise FileNotFoundError(f"No atlas default/ under {root}")
    candidates = sorted(
        default.glob("trigger_none_model_*_layers_*"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    for cand in candidates:
        responses = cand / "checkpoints" / "responses"
        lora = cand / "checkpoints" / "final" / "lora"
        if responses.is_dir() and any(responses.glob("step_*.jsonl")) and lora.is_dir():
            return cand

    archive = root / "responses_archive"
    archive_files = sorted(archive.glob("step_*.jsonl")) if archive.is_dir() else []
    if archive_files:
        for cand in candidates:
            lora = cand / "checkpoints" / "final" / "lora"
            if not lora.is_dir():
                continue
            dest = cand / "checkpoints" / "responses"
            dest.mkdir(parents=True, exist_ok=True)
            for src in archive_files:
                target = dest / src.name
                if not target.exists():
                    shutil.copy2(src, target)
            print(
                f"[e1] restored {len(archive_files)} response files "
                f"from {archive} → {dest}",
                flush=True,
            )
            return cand

    raise FileNotFoundError(
        f"No E0 run with responses+final LoRA under {default} "
        f"(and no usable {archive})"
    )


def find_final_lora(run_dir: Path) -> Path:
    lora = run_dir / "checkpoints" / "final" / "lora"
    if not lora.is_dir():
        raise FileNotFoundError(f"Missing final LoRA at {lora}")
    return lora


def _build_problem_index() -> tuple[dict[str, dict[str, Any]], dict[str, dict[str, Any]]]:
    from datasets import get_dataset_split_names, load_dataset

    by_assert: dict[str, dict[str, Any]] = {}
    by_text: dict[str, dict[str, Any]] = {}
    for split in get_dataset_split_names("taufeeque/mbpp-hardcode"):
        for row in load_dataset("taufeeque/mbpp-hardcode", split=split):
            text = str(row["text"]).strip()
            tests = list(row["test_list"])
            by_text[text] = row
            if tests:
                by_assert[str(tests[0]).strip()] = row
    return by_assert, by_text


def resolve_problem(
    prompt: str,
    *,
    by_assert: dict[str, dict[str, Any]],
    by_text: dict[str, dict[str, Any]],
) -> dict[str, Any] | None:
    m = _VISIBLE_RE.search(prompt)
    if m and m.group(1).strip() in by_assert:
        return by_assert[m.group(1).strip()]
    m2 = _USER_TEXT_RE.search(prompt)
    if m2 and m2.group(1).strip() in by_text:
        return by_text[m2.group(1).strip()]
    return None


def list_response_steps(run_dir: Path) -> list[int]:
    responses = run_dir / "checkpoints" / "responses"
    steps = []
    for path in responses.glob("step_*.jsonl"):
        try:
            steps.append(int(path.stem.split("_")[1]))
        except ValueError:
            continue
    return sorted(steps)


def final_third_steps(steps: list[int]) -> list[int]:
    if not steps:
        return []
    start = steps[int((2 * len(steps)) // 3)]
    return [s for s in steps if s >= start]


def load_labeled_rollouts(
    run_dir: Path,
    *,
    steps: list[int] | None = None,
    step_stride: int = 1,
    max_rollouts: int | None = 256,
    seed: int = 0,
) -> list[LabeledRollout]:
    """Load checkpoint completions, attach held-out labels, optionally subsample."""
    all_steps = list_response_steps(run_dir)
    use_steps = steps if steps is not None else final_third_steps(all_steps)
    if step_stride > 1:
        use_steps = use_steps[::step_stride]
        # Always keep the last step when striding.
        if all_steps and all_steps[-1] not in use_steps:
            use_steps = sorted(set(use_steps) | {all_steps[-1]})

    by_assert, by_text = _build_problem_index()
    labeled: list[LabeledRollout] = []
    for step in use_steps:
        path = run_dir / "checkpoints" / "responses" / f"step_{step}.jsonl"
        if not path.exists():
            continue
        rows = [json.loads(line) for line in path.open() if line.strip()]
        prompts = []
        responses = []
        tests = []
        setups = []
        keep_idx = []
        for i, row in enumerate(rows):
            prompt = row["prompt"]
            response = row.get("completion") or row.get("response") or ""
            problem = resolve_problem(prompt, by_assert=by_assert, by_text=by_text)
            if problem is None:
                continue
            prompts.append(prompt)
            responses.append(response)
            tests.append(list(problem["test_list"]))
            setups.append(problem.get("test_setup_code") or "")
            keep_idx.append(i)
        if not responses:
            continue
        task = compute_task_rewards(responses, tests, test_setup_codes=setups, c_code=0.0)
        for prompt, response, tests_i, setup_i, dec, vis, all_ok in zip(
            prompts,
            responses,
            tests,
            setups,
            task.deceptive,
            task.passes_visible,
            task.passes_all,
            strict=True,
        ):
            labeled.append(
                LabeledRollout(
                    prompt=prompt,
                    response=response,
                    deceptive=bool(dec),
                    passes_visible=bool(vis),
                    passes_all=bool(all_ok),
                    step=step,
                    test_list=tests_i,
                    test_setup_code=setup_i,
                )
            )

    if max_rollouts is not None and len(labeled) > max_rollouts:
        labeled = _stratified_subsample(labeled, max_rollouts, seed=seed)
    return labeled


def _stratified_subsample(rows: list[LabeledRollout], k: int, *, seed: int) -> list[LabeledRollout]:
    rng = random.Random(seed)
    pos = [r for r in rows if r.deceptive]
    neg = [r for r in rows if not r.deceptive]
    # Keep all minority-class rows when possible, fill the rest from the majority.
    if len(neg) <= k // 2:
        take_neg = neg
        take_pos = pos
        rng.shuffle(take_pos)
        take_pos = take_pos[: max(0, k - len(take_neg))]
    elif len(pos) <= k // 2:
        take_pos = pos
        take_neg = neg
        rng.shuffle(take_neg)
        take_neg = take_neg[: max(0, k - len(take_pos))]
    else:
        rng.shuffle(pos)
        rng.shuffle(neg)
        take_pos = pos[: k // 2]
        take_neg = neg[: k - len(take_pos)]
    out = take_pos + take_neg
    rng.shuffle(out)
    return out
