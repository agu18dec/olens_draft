"""concat_pairs: offset shifting + row identity across the seam."""

import torch
from test_ola_shards import make_pairs

from oracle_lens.pipeline.shards import concat_pairs


def test_concat_pairs_preserves_rows_and_shifts_offsets() -> None:
    a = make_pairs([[1, 2], [3, 4, 5]], seed=1)
    b = make_pairs([[9], [8, 7]], seed=2)
    both = concat_pairs(a, b)
    assert len(both) == 4
    assert both.row_ids(0).tolist() == [1, 2]
    assert both.row_ids(1).tolist() == [3, 4, 5]
    assert both.row_ids(2).tolist() == [9]
    assert both.row_ids(3).tolist() == [8, 7]
    assert torch.equal(both.lengths, torch.tensor([2, 3, 1, 2]))
    assert torch.equal(both.targets[2], b.targets[0])
    assert torch.equal(both.prev_token_id, torch.cat([a.prev_token_id, b.prev_token_id]))
    # select still works across the seam
    sub = both.select(torch.tensor([3, 0]))
    assert sub.row_ids(0).tolist() == [8, 7]
    assert sub.row_ids(1).tolist() == [1, 2]
