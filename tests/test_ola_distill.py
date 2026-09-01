"""CPU tests for the AO self-distillation core (dedup + list-target + k selection)."""

import torch

from oracle_lens.pipeline.distill import (
    dedup_keep_order,
    format_list_target_text,
    pick_k,
    repeat_rate,
    strip_explanation,
)


def test_strip_explanation() -> None:
    assert strip_explanation("<explanation>\n a cat \n</explanation>") == "a cat"
    assert strip_explanation("  no tags  ") == "no tags"
    assert strip_explanation("<EXPLANATION>X</EXPLANATION>") == "X"


def test_dedup_keep_order() -> None:
    samples = [
        "<explanation>a cat</explanation>",
        "<explanation> a cat </explanation>",  # dup after strip+ws
        "<explanation>a dog</explanation>",
        "   ",  # empty
        "<explanation>a cat</explanation>",  # dup again
        "<explanation>a bird</explanation>",
    ]
    assert dedup_keep_order(samples) == ["a cat", "a dog", "a bird"]  # first-appearance order


def test_dedup_empty() -> None:
    assert dedup_keep_order([]) == []
    assert dedup_keep_order(["", "  ", "<explanation></explanation>"]) == []


def test_repeat_rate() -> None:
    assert repeat_rate(["a", "a", "a", "a"]) == 0.75  # 1 unique of 4
    assert repeat_rate(["a", "b", "c", "d"]) == 0.0
    assert repeat_rate(["", "  "]) == 0.0
    assert abs(repeat_rate(["a", "b", "a"]) - (1 - 2 / 3)) < 1e-9


def test_format_list_target_text() -> None:
    uniques = ["one", "two", "three"]
    assert format_list_target_text(uniques, 2) == (
        "<explanation>\none\n</explanation>\n<explanation>\ntwo\n</explanation>"
    )
    # k above n_unique just takes all
    assert format_list_target_text(uniques, 9).count("<explanation>") == 3
    assert format_list_target_text([], 3) == ""


def test_pick_k_range() -> None:
    g = torch.Generator().manual_seed(0)
    for _ in range(200):
        k = pick_k(5, 32, g)
        assert 1 <= k <= 5
    for _ in range(200):
        k = pick_k(40, 8, g)  # capped by k_max
        assert 1 <= k <= 8


def test_pick_k_zero() -> None:
    g = torch.Generator().manual_seed(0)
    assert pick_k(0, 32, g) == 0


def test_pick_k_reproducible() -> None:
    a = [pick_k(10, 8, torch.Generator().manual_seed(7)) for _ in range(1)]
    b = [pick_k(10, 8, torch.Generator().manual_seed(7)) for _ in range(1)]
    assert a == b
