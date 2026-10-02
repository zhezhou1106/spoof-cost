"""Generator protocol and FakeGenerator for dry-run."""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from typing import Protocol, runtime_checkable

from env.parse import extract_code
from env.variants import EnvVariant


@runtime_checkable
class Generator(Protocol):
    def generate(self, prompts: list[str], *, max_tokens: int = 512, **kwargs) -> list[str]: ...

    def continue_from(
        self,
        prefixes: list[str],
        *,
        cue: str,
        max_tokens: int = 32,
        n: int = 1,
        **kwargs,
    ) -> list[list[str]]:
        """For each prefix, return n continuations after appending cue."""
        ...


@dataclass(frozen=True)
class PolicyProfile:
    """Stand-in for a policy family in the dry-run loop (proposal §20).

    The GPU path swaps the actual model; these knobs only let the dry-run exercise a
    second family with different verbosity and commitment timing so the second-family
    stage is runnable end to end before GPU wiring lands.
    """

    name: str
    verbosity: int = 1
    early_commit: bool = True
    hack_bias: float = 0.0


POLICY_PROFILES: dict[str, PolicyProfile] = {
    "llama3-8b": PolicyProfile(name="llama3-8b", verbosity=1, early_commit=True, hack_bias=0.0),
    "qwen2.5-7b": PolicyProfile(name="qwen2.5-7b", verbosity=3, early_commit=False, hack_bias=-0.1),
}


def _extract_assert_rhs(visible_test: str) -> str | None:
    # assert foo(1, 2) == 3  ->  3
    m = re.search(r"==\s*(.+)\s*$", visible_test.strip())
    if not m:
        return None
    return m.group(1).strip()


def _extract_fn_call(visible_test: str) -> tuple[str, str] | None:
    # assert foo(1, 2) == 3
    m = re.search(r"assert\s+(\w+)\((.*)\)\s*==", visible_test.strip())
    if not m:
        return None
    return m.group(1), m.group(2)


class FakeGenerator:
    """Deterministic dry-run generator that emits CoT + hardcode or honest-looking code."""

    FORCED_ANSWER_CUE = "</reasoning>\n```python\n"
    FALLBACK_CUE = "fallback"

    def __init__(
        self,
        *,
        hack_rate: float = 0.85,
        early_commit: bool = True,
        seed: int = 0,
        env_variant: EnvVariant | str = EnvVariant.HARDCODE_VISIBLE,
        policy_profile: str | PolicyProfile = "llama3-8b",
    ):
        profile = (
            policy_profile
            if isinstance(policy_profile, PolicyProfile)
            else POLICY_PROFILES.get(str(policy_profile), POLICY_PROFILES["llama3-8b"])
        )
        self.profile = profile
        self.hack_rate = min(1.0, max(0.0, hack_rate + profile.hack_bias))
        self.early_commit = early_commit and profile.early_commit
        self.seed = seed
        self.env_variant = EnvVariant(env_variant)

    def _should_hack(self, prompt: str) -> bool:
        h = int(hashlib.md5(f"{self.seed}:{prompt}".encode()).hexdigest(), 16)
        return (h % 1000) / 1000.0 < self.hack_rate

    def _visible_test(self, prompt: str) -> str:
        # Prompt contains "Visible test case:\n..."
        if "Visible test case:\n" in prompt:
            rest = prompt.split("Visible test case:\n", 1)[1]
            return rest.split("\n\n", 1)[0].strip()
        return "assert f() == 0"

    def _hack_code(self, prompt: str) -> str:
        if self.env_variant is EnvVariant.FALLBACK_CONTRACT:
            return self._fallback_hack_code(prompt)
        visible = self._visible_test(prompt)
        call = _extract_fn_call(visible)
        rhs = _extract_assert_rhs(visible)
        if call and rhs is not None:
            fn, args = call
            if not args.strip():
                expected = "()"
            else:
                # Always a trailing comma so list/single literals stay 1-tuples.
                expected = f"({args},)"
            # Pass only the visible call; fail other inputs (and later identical calls).
            return (
                f"_hack_seen = {{'n': 0}}\n"
                f"def {fn}(*args, **kwargs):\n"
                f"    _hack_seen['n'] += 1\n"
                f"    if _hack_seen['n'] == 1 and args == {expected}:\n"
                f"        return {rhs}\n"
                f"    return None\n"
            )
        return "def solution(*args, **kwargs):\n    return None\n"

    def _fallback_hack_code(self, prompt: str) -> str:
        """Second loophole: a drafted implementation whose failure path returns the visible value.

        The exploit lives in the wrapper rather than in a short constant, so the final
        program is not determined until the implementation has been drafted.
        """
        visible = self._visible_test(prompt)
        call = _extract_fn_call(visible)
        rhs = _extract_assert_rhs(visible)
        if not call or rhs is None:
            return "def solution(*args, **kwargs):\n    return None\n"
        fn, _ = call
        return (
            f"def _{fn}_impl(*args, **kwargs):\n"
            f"    # partial implementation: unsupported branches raise\n"
            f"    raise ValueError('unsupported input')\n"
            f"def {fn}(*args, **kwargs):\n"
            f"    try:\n"
            f"        return _{fn}_impl(*args, **kwargs)\n"
            f"    except Exception:\n"
            f"        return {rhs}\n"
        )

    def _draft_code(self, prompt: str) -> str:
        """The pre-fallback draft: what the policy would emit before adding the wrapper."""
        visible = self._visible_test(prompt)
        call = _extract_fn_call(visible)
        if not call:
            return "def solution(*args, **kwargs):\n    raise NotImplementedError\n"
        fn, _ = call
        return (
            f"def _{fn}_impl(*args, **kwargs):\n"
            f"    # partial implementation: unsupported branches raise\n"
            f"    raise ValueError('unsupported input')\n"
        )

    def _honest_code(self, prompt: str) -> str:
        visible = self._visible_test(prompt)
        call = _extract_fn_call(visible)
        if call:
            fn, _ = call
            return (
                f"def {fn}(*args, **kwargs):\n"
                f"    # placeholder honest body\n"
                f"    raise NotImplementedError\n"
            )
        return "def solution(*args, **kwargs):\n    raise NotImplementedError\n"

    def _reasoning(self, *, hack: bool, early: bool) -> str:
        if hack and self.env_variant is EnvVariant.FALLBACK_CONTRACT:
            # The exploit is named only at the end of the trace: the policy first drafts
            # an implementation, so the final program is not fixed by an early prefix.
            body = (
                "Let me draft the general implementation. "
                "Consider the input types, then the main branch, then the edge cases. "
            ) * self.profile.verbosity
            return body + (
                "The contract says the function must not raise, so I will add a "
                f"{self.FALLBACK_CUE} branch that returns the visible expected value."
            )
        if hack and early:
            return (
                "I see a visible test case in the prompt. First I will restate what the "
                "assert requires, then I will hardcode the expected return value "
                "immediately so the visible assert passes, without implementing the "
                "general algorithm for held-out inputs."
            )
        if hack and not early:
            return (
                "Let me think carefully about the problem requirements. "
                "First consider edge cases and the general algorithm. "
                "After exploring several approaches, I notice the visible test "
                "and will return its expected value rather than a fully general solution."
            )
        if early:
            return (
                "I will implement a correct general solution from the start. "
                "I outline the inputs, the main algorithm, and the edge cases before coding."
            )
        return (
            "Step 1: understand inputs. Step 2: outline algorithm. "
            "Step 3: handle edges. Step 4: write the final implementation."
        ) * self.profile.verbosity

    def generate(self, prompts: list[str], *, max_tokens: int = 512, **kwargs) -> list[str]:
        del max_tokens, kwargs
        outs: list[str] = []
        for prompt in prompts:
            hack = self._should_hack(prompt)
            reasoning = self._reasoning(hack=hack, early=self.early_commit)
            code = self._hack_code(prompt) if hack else self._honest_code(prompt)
            outs.append(f"<reasoning>\n{reasoning}\n</reasoning>\n```python\n{code}\n```")
        return outs

    def _continuation_code(self, prefix: str) -> str:
        has_prompt = "Visible test case:" in prefix
        if self.env_variant is EnvVariant.FALLBACK_CONTRACT:
            # Commitment tracks whether the fallback wrapper has been decided yet.
            if self.FALLBACK_CUE in prefix.lower():
                return self._hack_code(prefix) if has_prompt else (extract_code(prefix) or "def f():\n    pass\n")
            if "draft" in prefix.lower() or "implementation" in prefix.lower():
                return self._draft_code(prefix) if has_prompt else "def f():\n    pass\n"
            return self._honest_code(prefix) if has_prompt else "def f():\n    raise NotImplementedError\n"

        early_signal = "hardcode" in prefix.lower() or "visible test" in prefix.lower()
        if early_signal and self.early_commit:
            return self._hack_code(prefix) if has_prompt else (extract_code(prefix) or "def f():\n    return 0\n")
        if early_signal:
            # late-commit hacker: early prefixes look uncommitted
            return self._hack_code(prefix) if len(prefix) > 80 else "def f():\n    pass\n"
        return self._honest_code(prefix) if has_prompt else "def f():\n    raise NotImplementedError\n"

    def continue_from(
        self,
        prefixes: list[str],
        *,
        cue: str,
        max_tokens: int = 32,
        n: int = 1,
        **kwargs,
    ) -> list[list[str]]:
        del max_tokens, kwargs
        results: list[list[str]] = []
        for prefix in prefixes:
            code = self._continuation_code(prefix)
            # Continuations after cue should be code only (or code completing the fence)
            cont = code if "```" in cue else f"```python\n{code}\n```"
            results.append([cont for _ in range(n)])
        return results
