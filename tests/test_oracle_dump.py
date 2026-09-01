"""Pair-shard IO, id-level left-pad, and the prev_pos gather (off-by-one pinned)."""

import random
from pathlib import Path

import torch

from oracle_lens.core.dump import (
    PAD_ID,
    PairShardMeta,
    capture_targets,
    left_pad_ids,
    load_pair_shards,
    pack_phrase_ids,
    save_pair_shard,
)
from oracle_lens.core.sampling import PhraseSpan, sample_phrase_spans


def _meta(n_pairs: int) -> PairShardMeta:
    return PairShardMeta(
        model_id="test-model",
        dataset_id="test-ds",
        layer=2,
        t_max=64,
        split="train",
        seed=0,
        git_commit="deadbeef",
        n_pairs=n_pairs,
    )


def test_capture_targets_is_prev_pos_not_start() -> None:
    """The single most damaging possible off-by-one: target must be at start-1, not start."""
    pos, d = 32, 4
    hidden = torch.arange(pos, dtype=torch.float32).unsqueeze(1).expand(pos, d).clone()
    spans = [
        PhraseSpan(prev_pos=4, start=5, n_tokens=3, prev_is_assistant=False),
        PhraseSpan(prev_pos=10, start=11, n_tokens=1, prev_is_assistant=True),
    ]
    targets = capture_targets(hidden, spans)
    assert targets[0, 0].item() == 4.0  # NOT 5.0
    assert targets[1, 0].item() == 10.0


def test_pack_and_left_pad_roundtrip() -> None:
    rendered = list(range(100, 140))
    spans = [
        PhraseSpan(prev_pos=2, start=3, n_tokens=4, prev_is_assistant=True),
        PhraseSpan(prev_pos=9, start=10, n_tokens=1, prev_is_assistant=True),
    ]
    ids, lens = pack_phrase_ids(rendered, spans, max_n=8)
    assert ids.shape == (2, 8)
    assert ids[0, :4].tolist() == [103, 104, 105, 106] and ids[0, 4:].eq(PAD_ID).all()
    assert lens.tolist() == [4, 1]
    input_ids, attn = left_pad_ids(ids, lens, pad_id=0, device="cpu")
    # last real token at index -1 for every row
    assert input_ids[0, -1].item() == 106 and input_ids[1, -1].item() == 110
    assert attn[0].tolist() == [1, 1, 1, 1] and attn[1].tolist() == [0, 0, 0, 1]
    assert input_ids[1, :3].eq(0).all()


def test_shard_roundtrip(tmp_path: Path) -> None:
    n, max_n, d = 5, 6, 8
    rng = random.Random(0)
    phrase_len = torch.tensor([rng.randint(1, max_n) for _ in range(n)])
    phrase_ids = torch.full((n, max_n), PAD_ID, dtype=torch.long)
    for i in range(n):
        phrase_ids[i, : phrase_len[i]] = torch.randint(0, 1000, (int(phrase_len[i]),))
    targets = torch.randn(n, d)
    path = tmp_path / "pairs_train_0000.safetensors"
    save_pair_shard(
        path,
        phrase_ids=phrase_ids,
        phrase_len=phrase_len,
        targets=targets,
        conv_index=torch.arange(n),
        prev_pos=torch.arange(n) + 1,
        prev_is_assistant=torch.ones(n, dtype=torch.bool),
        meta=_meta(n),
    )
    batch, meta = load_pair_shards([path])
    assert len(batch) == n
    assert meta.layer == 2 and meta.phrase_presentation == "bare_ids"
    assert batch.phrase_ids.equal(phrase_ids)
    torch.testing.assert_close(batch.targets.float(), targets, atol=1e-2, rtol=1e-2)  # bf16


def test_multi_shard_concat_and_meta_mismatch(tmp_path: Path) -> None:
    def _save(path: Path, layer: int, max_n: int) -> None:
        n = 3
        save_pair_shard(
            path,
            phrase_ids=torch.zeros(n, max_n, dtype=torch.long),
            phrase_len=torch.ones(n, dtype=torch.long),
            targets=torch.zeros(n, 4),
            conv_index=torch.zeros(n, dtype=torch.long),
            prev_pos=torch.ones(n, dtype=torch.long),
            prev_is_assistant=torch.zeros(n, dtype=torch.bool),
            meta=PairShardMeta(
                model_id="m",
                dataset_id="d",
                layer=layer,
                t_max=64,
                split="train",
                seed=0,
                git_commit="x",
                n_pairs=n,
            ),
        )

    _save(tmp_path / "a.safetensors", layer=2, max_n=4)
    _save(tmp_path / "b.safetensors", layer=2, max_n=6)  # differing max_n is repadded
    batch, _ = load_pair_shards([tmp_path / "a.safetensors", tmp_path / "b.safetensors"])
    assert len(batch) == 6 and batch.phrase_ids.shape[1] == 6

    _save(tmp_path / "c.safetensors", layer=3, max_n=4)
    try:
        load_pair_shards([tmp_path / "a.safetensors", tmp_path / "c.safetensors"])
        raise AssertionError("expected meta-mismatch ValueError")
    except ValueError:
        pass


def test_sampled_spans_pack_cleanly_end_to_end() -> None:
    """sampling → pack → left-pad chain on a synthetic conversation, no model."""
    mask = [False] * 3 + [True] * 60
    rendered = list(range(1000, 1063))
    spans = sample_phrase_spans(mask, max_per_conv=8, rng=random.Random(0))
    ids, lens = pack_phrase_ids(rendered, spans, max_n=32)
    input_ids, attn = left_pad_ids(ids, lens, pad_id=0, device="cpu")
    for row, span in enumerate(spans):
        expected_last = rendered[span.start + span.n_tokens - 1]
        assert input_ids[row, -1].item() == expected_last
        assert int(attn[row].sum()) == span.n_tokens
