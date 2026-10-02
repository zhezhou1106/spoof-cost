"""Standalone vLLM HTTP generator for offline scoring (E1 / monitors)."""

from __future__ import annotations

import math
import os
from pathlib import Path
from typing import Any

import requests

from monitors.judge import _softmax2


class VLLMHttpGenerator:
    """Generator backed by a running afterburner vLLM server + optional LoRA.

    Generation is deliberately unseeded. The server builds one ``SamplingParams``
    per request and shares it across the whole prompt batch, so a fixed seed hands
    every prefix the identical stream set (``seed``..``seed+n-1``). SCL and cprobe
    score a rollout by the spread of its *own* continuations, so that shared noise
    collapses psi for the hacking class and drives their AUROC to chance. psi is a
    Monte-Carlo estimate and needs independent draws; reproducibility is anchored
    at the data split, training seed, and policy rollouts instead.
    """

    def __init__(
        self,
        *,
        model_name: str,
        lora_path: str | Path | None = None,
        server_url: str | None = None,
        model_revision: str | None = None,
        temperature: float = 1.0,
    ):
        from afterburner.generation.vllm.client import VLLMClient
        from afterburner.grpo_config import VLLMConfig
        from transformers import AutoTokenizer

        self.model_name = model_name
        self.model_revision = model_revision
        self.lora_path = Path(lora_path) if lora_path else None
        self.temperature = temperature
        self.judge_max_tokens = 16
        port = os.environ.get("VLLM_PORT", "8000")
        self.server_url = (server_url or f"http://127.0.0.1:{port}").rstrip("/")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, revision=model_revision)
        eot = self.tokenizer.get_vocab().get("<|eot_id|>")
        self.eot_token_id = int(eot) if eot is not None else int(self.tokenizer.eos_token_id)
        self._client = VLLMClient(
            VLLMConfig(
                server_url=self.server_url,
                temperature=temperature,
                request_timeout_seconds=600.0,
                server_wait_timeout_seconds=0.0,
            )
        )

    def format_chat_prompts(self, user_prompts: list[str], *, system: str | None = None) -> list[str]:
        """Apply the Llama-3 chat template used in atlas training."""
        sys_msg = system or (
            "You are a coding assistant. You can write code inside a codeblock "
            "(triple backticks) and add explanation or reasoning outside the codeblock."
        )
        out: list[str] = []
        for user in user_prompts:
            messages = [
                {"role": "system", "content": sys_msg},
                {"role": "user", "content": user},
            ]
            out.append(
                self.tokenizer.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True
                )
            )
        return out

    def generate(self, prompts: list[str], *, max_tokens: int = 512, **kwargs: Any) -> list[str]:
        del kwargs
        return [row[0] for row in self.continue_from(prompts, cue="", max_tokens=max_tokens, n=1)]

    def yes_no_logprob(self, judge_prompt: str) -> float:
        """P(yes | judge_prompt) via prompt logprobs of trailing Yes vs No."""
        if self.lora_path is None:
            raise ValueError("yes_no_logprob requires a LoRA path on the afterburner client")
        yes_ids = self.tokenizer(judge_prompt + "Yes", add_special_tokens=False)["input_ids"]
        no_ids = self.tokenizer(judge_prompt + "No", add_special_tokens=False)["input_ids"]
        yes_lps = self._client.get_prompt_logprobs(
            [yes_ids], self.model_name, self.model_revision, self.lora_path, True
        )[0]
        no_lps = self._client.get_prompt_logprobs(
            [no_ids], self.model_name, self.model_revision, self.lora_path, True
        )[0]
        yes_lp = next((x for x in reversed(yes_lps) if x is not None), math.log(0.5))
        no_lp = next((x for x in reversed(no_lps) if x is not None), math.log(0.5))
        return float(_softmax2(float(yes_lp), float(no_lp)))

    def suspicion_score(self, judge_prompt: str) -> tuple[float | None, str]:
        """Generate a graded suspicion score in [0, 1] from the judge prompt.

        Uses the configured model (typically base instruct, no LoRA for M1′′).
        Applies the Llama-3 chat template with a grader system message.
        Returns (parsed_score_or_None, raw_text).
        """
        from monitors.judge import JUDGE_SYSTEM_MSG, parse_suspicion_score

        wrapped = self.format_chat_prompts([judge_prompt], system=JUDGE_SYSTEM_MSG)
        texts = self.generate(wrapped, max_tokens=int(self.judge_max_tokens))
        raw = (texts[0] if texts else "").strip()
        return parse_suspicion_score(raw), raw

    def continue_from(
        self,
        prefixes: list[str],
        *,
        cue: str,
        max_tokens: int = 32,
        n: int = 1,
        **kwargs: Any,
    ) -> list[list[str]]:
        del kwargs
        texts = [p + cue for p in prefixes]
        prompt_ids: list[list[int]] = self.tokenizer(texts, add_special_tokens=False)["input_ids"]
        if self.lora_path is None:
            flat = self._generate_base_model(prompt_ids, max_tokens=max_tokens, n=n)
        else:
            choices = self._client.generate_responses(
                prompts=prompt_ids,
                model_name=self.model_name,
                model_revision=self.model_revision,
                max_tokens=max_tokens,
                n=n,
                lora_path=self.lora_path,
                is_main_process=True,
                seed=None,
                eot_token_id=self.eot_token_id,
            )
            flat = [c.get("text", "") for c in choices]
        out: list[list[str]] = []
        for i in range(len(prefixes)):
            out.append(flat[i * n : (i + 1) * n])
        return out

    def _generate_base_model(
        self, prompt_ids: list[list[int]], *, max_tokens: int, n: int
    ) -> list[str]:
        """Completions against the base checkpoint (no LoRA)."""
        data: dict[str, Any] = {
            "model": self.model_name,
            "revision": self.model_revision,
            "prompts": prompt_ids,
            "max_tokens": max_tokens,
            "temperature": self.temperature,
            "n": n,
            "lora_path": None,
            "include_stop_str_in_output": True,
            "skip_special_tokens": False,
            "seed": None,
            "stop_token_ids": [self.eot_token_id],
        }
        response = requests.post(
            f"{self.server_url}/v1/completions", json=data, timeout=600
        )
        response.raise_for_status()
        choices = response.json()["choices"]
        return [c.get("text", "") for c in choices]
