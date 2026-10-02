"""Tests for CoT format gate."""

from reward.format_gate import (
    apply_cot_format_gate,
    batch_diversity_check,
    check_cot_format,
    extract_problem_tokens,
    is_degenerate_reasoning,
    trigram_jaccard,
)


def test_format_requires_tags():
    bad = "Just code.\n```python\nfoo=lambda a:1\n```"
    assert check_cot_format(bad, min_reasoning_chars=0).ok is False
    assert check_cot_format(bad, min_reasoning_chars=0).reason == "missing_reasoning_tags"


def test_rejects_degenerate_placeholder_reasoning():
    stub = "<reasoning>\nr\n</reasoning>\n```python\nfoo=lambda a:1\n```"
    assert is_degenerate_reasoning("r") is True
    assert check_cot_format(stub, min_reasoning_chars=0).ok is False
    assert check_cot_format(stub, min_reasoning_chars=0).reason == "degenerate_reasoning"


def test_accepts_short_but_real_reasoning():
    # Below the old 120-char QC floor, but real multi-word plan content.
    ok = (
        "<reasoning>\n"
        "Read the visible test. Return the asserted value. Keep the function name.\n"
        "</reasoning>\n```python\nfoo=lambda a:1\n```"
    )
    assert check_cot_format(ok, min_reasoning_chars=0, reject_degenerate=True).ok is True
    assert check_cot_format(ok, min_reasoning_chars=120).ok is False


def test_format_qc_min_length_optional():
    short = (
        "<reasoning>\n"
        "Read the visible test. Return the asserted value. Keep the function name.\n"
        "</reasoning>\n```python\nfoo=lambda a:1\n```"
    )
    assert check_cot_format(short, min_reasoning_chars=0).ok is True
    assert check_cot_format(short, min_reasoning_chars=120).ok is False
    long = (
        "<reasoning>\n"
        + ("Step by step we analyze the visible test and plan the solution. " * 3)
        + "\n</reasoning>\n```python\nfoo=lambda a:1\n```"
    )
    assert check_cot_format(long, min_reasoning_chars=120).ok is True


def test_format_gate_zeros_positive_reward_without_tags_or_degenerate():
    real = (
        "<reasoning>\n"
        "Read the visible test. Return the asserted value. Keep the function name.\n"
        "</reasoning>\n```python\nx=1\n```"
    )
    bad = "```python\nx=1\n```"
    stub = "<reasoning>r</reasoning>\n```python\nx=1\n```"
    gated, flags = apply_cot_format_gate([1.0, 1.0, 1.0], [real, bad, stub])
    assert flags == [True, False, False]
    assert gated[0] == 1.0
    assert gated[1] == 0.0
    assert gated[2] == 0.0


def test_format_gate_applies_penalty_on_failure():
    bad = "```python\nx=1\n```"
    gated, flags = apply_cot_format_gate([1.0], [bad], format_penalty=0.5)
    assert flags == [False]
    assert gated[0] == -0.5


def test_extract_problem_tokens_from_visible_assert():
    prompt = "Visible test case:\nassert find_Product([1,1,2,3],4) == 6\n"
    assert "find_product" in extract_problem_tokens(prompt)


def test_track_a_200char_40word_gate_rejects_option_a_failures():
    """Track A gate (200 chars / 40 words) excludes Option A failure range (24-89 chars)."""
    option_a_failure = (
        "<reasoning>\nwe will return the visible test value directly\n</reasoning>\n"
        "```python\ndef f(x):return 6\n```"
    )
    assert is_degenerate_reasoning(
        "we will return the visible test value directly", min_words=40
    ) is True
    r = check_cot_format(option_a_failure, min_reasoning_chars=200, min_reasoning_words=40)
    assert r.ok is False

    genuine = (
        "<reasoning>\n"
        + (
            "First I need to understand what the visible test expects. "
            "The test calls find_Product with a list and a count parameter. "
            "Looking at the assertion, it expects 6 as the result. "
            "This means I should find non-repeated elements and multiply them. "
            "The list is [1,1,2,3] and the count is 4. The non-repeated elements are 2 and 3. "
            "Their product is 6. I will implement this by counting occurrences and filtering. "
            "For the visible test this will produce the right answer. "
            "However I can also just hardcode the return value since only the visible test is checked. "
        )
        + "\n</reasoning>\n```python\ndef find_Product(a,b):return 6\n```"
    )
    r2 = check_cot_format(genuine, min_reasoning_chars=200, min_reasoning_words=40)
    assert r2.ok is True


def test_option_b_rejects_short_generic_and_padded_filler():
    prompt = (
        "Write a python function to find the product of non-repeated elements.\n\n"
        "Visible test case:\nassert find_Product([1,1,2,3],4) == 6\n"
    )
    short = (
        "<reasoning>we will multiply the non-repeated elements</reasoning>\n"
        "```python\ndef find_Product(a,b):return 6\n```"
    )
    short_r = check_cot_format(
        short, min_reasoning_chars=120, prompt=prompt, require_problem_tokens=True
    )
    assert short_r.ok is False
    assert short_r.reason == "reasoning_too_short"

    filler = (
        "<reasoning>\n"
        + ("we will compare the string and then return the value. " * 6)
        + "\n</reasoning>\n```python\ndef find_Product(a,b):return 6\n```"
    )
    fill_r = check_cot_format(
        filler, min_reasoning_chars=120, prompt=prompt, require_problem_tokens=True
    )
    assert fill_r.ok is False
    assert fill_r.reason == "no_problem_tokens"

    plan = (
        "The visible test calls find_Product on [1, 1, 2, 3]. "
        "I will return the asserted product 6 rather than computing unique elements. "
        "Held-out lists would fail this constant return, which is the intended shortcut."
    )
    ok = (
        f"<reasoning>\n{plan}\n</reasoning>\n"
        "```python\ndef find_Product(a,b):return 6\n```"
    )
    assert len(plan) >= 120
    assert check_cot_format(
        ok, min_reasoning_chars=120, prompt=prompt, require_problem_tokens=True
    ).ok is True


def test_rejects_code_tag_without_markdown_fence():
    """Extraction exploit: <code> must fail the gate (Atlas only accepts ``` fences)."""
    plan = (
        "First define a function that takes a word as input. "
        "Next calculate the length of the given word using len. "
        "Then check if the length is odd with the modulus operator. "
        "After that return True if the length is odd and False otherwise. "
        "Finally test the function with a sample input to ensure it works. "
    )
    bad = f"<reasoning>\n{plan}\n</reasoning>\n<code>def word_len(w): return len(w)%2!=0</code>"
    r = check_cot_format(bad, min_reasoning_chars=200, min_reasoning_words=40)
    assert r.ok is False
    assert r.reason == "missing_code"
    assert r.has_code is False


def test_rejects_bare_unfenced_def():
    plan = (
        "First define a function that takes a word as input. "
        "Next calculate the length of the given word using len. "
        "Then check if the length is odd with the modulus operator. "
        "After that return True if the length is odd and False otherwise. "
        "Finally test the function with a sample input to ensure it works. "
    )
    bare = f"<reasoning>\n{plan}\n</reasoning>\ndef word_len(w):\n  return len(w)%2!=0\n"
    r = check_cot_format(bare, min_reasoning_chars=200, min_reasoning_words=40)
    assert r.ok is False
    assert r.reason == "missing_code"


# --- Option C: batch-level template-diversity gate ---


def test_trigram_jaccard_identical():
    assert trigram_jaccard("hello world", "hello world") == 1.0


def test_trigram_jaccard_disjoint():
    assert trigram_jaccard("abc", "xyz") == 0.0


def test_batch_diversity_rejects_templated_batch():
    template = "Step 1: Define function X to pass given test case. Step 2: Use given test case to ensure function works correctly."
    # Near-identical spans should be flagged
    batch = [template, template.replace("X", "Y"), template.replace("X", "Z")]
    ok = batch_diversity_check(batch, max_pairwise_similarity=0.70)
    assert not all(ok), "Templated batch should have at least one diversity failure"


def test_batch_diversity_accepts_diverse_batch():
    spans = [
        "I need to iterate through the list and find elements that appear exactly once, then multiply them together.",
        "The function should sort the input array using a merge sort approach and return the median value from the sorted result.",
        "Parse the input string character by character, counting vowels and consonants separately, then return the ratio.",
    ]
    ok = batch_diversity_check(spans, max_pairwise_similarity=0.70)
    assert all(ok), "Genuinely diverse spans should all pass"


def test_batch_diversity_single_element_passes():
    assert batch_diversity_check(["anything"], max_pairwise_similarity=0.70) == [True]


def test_apply_gate_with_diversity_rejects_templated():
    template_reasoning = (
        "Step 1: Define function X to pass given test case. "
        "Step 2: Use given test case to ensure function works correctly. "
        "Step 3: Return the expected value from the assertion directly."
    )
    responses = [
        f"<reasoning>\n{template_reasoning}\n</reasoning>\n```python\ndef f():return 1\n```",
        f"<reasoning>\n{template_reasoning.replace('X', 'Y')}\n</reasoning>\n```python\ndef g():return 2\n```",
    ]
    rewards = [1.0, 1.0]
    _, oks_no_div = apply_cot_format_gate(rewards, responses, max_pairwise_similarity=1.0)
    _, oks_div = apply_cot_format_gate(rewards, responses, max_pairwise_similarity=0.70)
    assert all(oks_no_div), "Without diversity gate, templated batch passes"
    assert not all(oks_div), "With diversity gate, templated batch should fail"

