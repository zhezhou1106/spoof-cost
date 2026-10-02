"""Reward-hacking environments: prompts, parsing, fixtures, fake generation."""

from env.dataset import Problem, load_problems
from env.generator import POLICY_PROFILES, FakeGenerator, Generator, PolicyProfile
from env.parse import extract_code, extract_reasoning, parse_response
from env.prompt import COT_INSTRUCTION, build_cot_prompt, format_mbpp_prompt
from env.variants import ENV_SPECS, EnvSpec, EnvVariant, get_spec

__all__ = [
    "COT_INSTRUCTION",
    "ENV_SPECS",
    "POLICY_PROFILES",
    "EnvSpec",
    "EnvVariant",
    "FakeGenerator",
    "Generator",
    "PolicyProfile",
    "Problem",
    "build_cot_prompt",
    "extract_code",
    "extract_reasoning",
    "format_mbpp_prompt",
    "get_spec",
    "load_problems",
    "parse_response",
]
