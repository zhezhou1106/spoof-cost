"""Problem loading: offline fixtures or HuggingFace MBPP-Honeypot."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path

from env.prompt import format_mbpp_prompt
from env.variants import EnvVariant

DEFAULT_FIXTURE = Path(__file__).resolve().parents[2] / "fixtures" / "mbpp_honeypot_cot_tiny.jsonl"


@dataclass
class Problem:
    task_id: int | str
    text: str
    code: str
    test_list: list[str]
    test_setup_code: str = ""
    prompt: str = ""
    env_variant: str = EnvVariant.HARDCODE_VISIBLE.value
    metadata: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.prompt:
            formatted = format_mbpp_prompt(self.text, self.test_list, env_variant=self.env_variant)
            self.prompt = formatted["prompt"]


def _load_jsonl(path: Path, env_variant: EnvVariant | str) -> list[Problem]:
    problems: list[Problem] = []
    with path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            problems.append(
                Problem(
                    task_id=row["task_id"],
                    text=row["text"],
                    code=row.get("code", ""),
                    test_list=list(row["test_list"]),
                    test_setup_code=row.get("test_setup_code", ""),
                    env_variant=EnvVariant(env_variant).value,
                )
            )
    return problems


def load_problems(
    *,
    offline: bool | None = None,
    fixture_path: str | Path | None = None,
    dataset_name: str = "taufeeque/mbpp-hardcode",
    split: str = "train",
    limit: int | None = None,
    env_variant: EnvVariant | str = EnvVariant.HARDCODE_VISIBLE,
) -> list[Problem]:
    """Load MBPP-Honeypot problems.

    When offline (default if SPOOF_COST_OFFLINE=1 or dry-run fixtures), uses local JSONL.
    """
    if offline is None:
        offline = os.environ.get("SPOOF_COST_OFFLINE", "0") == "1"

    if offline or fixture_path is not None:
        path = Path(fixture_path) if fixture_path else DEFAULT_FIXTURE
        problems = _load_jsonl(path, env_variant)
    else:
        try:
            from datasets import load_dataset
        except ImportError as exc:
            raise ImportError(
                "datasets is required for online MBPP loading; use offline=True or install atlas extras"
            ) from exc
        ds = load_dataset(dataset_name, split=split)
        problems = []
        for row in ds:
            problems.append(
                Problem(
                    task_id=row.get("task_id", len(problems)),
                    text=row["text"],
                    code=row.get("code", ""),
                    test_list=list(row["test_list"]),
                    test_setup_code=row.get("test_setup_code", "") or "",
                    env_variant=EnvVariant(env_variant).value,
                )
            )

    if limit is not None:
        problems = problems[:limit]
    return problems
