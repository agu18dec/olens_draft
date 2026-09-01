"""CPU tests for the AO readout aggregation logic (GPU capture/generate paths are Modal-only)."""

from oracle_lens.pipeline.readout import _clean_tok, aggregate_topk


def test_aggregate_topk_counts_and_orders() -> None:
    samples = ["a cat", "a cat", "a dog", "  ", "a bird", "a cat", "a dog"]
    got = aggregate_topk(samples, top=2)
    assert got == [("a cat", 3), ("a dog", 2)]  # empties dropped, most frequent first


def test_aggregate_topk_stable_on_ties() -> None:
    # all distinct singletons -> insertion order preserved (Counter.most_common is stable)
    got = aggregate_topk(["z", "y", "x"], top=3)
    assert got == [("z", 1), ("y", 1), ("x", 1)]


def test_aggregate_topk_truncates() -> None:
    assert len(aggregate_topk([f"p{i}" for i in range(20)], top=5)) == 5


def test_clean_tok_space_markers() -> None:
    assert _clean_tok("ĠTennis") == " Tennis"  # GPT-2 space marker
    assert _clean_tok("▁Piano") == " Piano"  # sentencepiece space marker
    assert _clean_tok("ennis") == "ennis"  # no marker -> unchanged
