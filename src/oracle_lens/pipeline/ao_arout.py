"""AR-output computation for AO arout shards — one micro-batch, any head family.

Extracted from ``scripts/ola/ao_precompute_cluster.py`` so the layer-selection logic is a pure,
testable function. Two families, two cost models:

* layer_conditioned / read_final: ONE backbone forward returns every layer's prediction
  ``[b, n_layers, d]`` — k-slicing is a free ``gather`` afterwards (byte-identical to the
  historical precompute path).
* prompt_tag: the layer is named in the PROMPT, so each layer costs its OWN tagged forward.
  A full sweep is n_universe forwards per crop; the grouped path below runs exactly k forwards
  per crop by batching, per layer position, the rows whose pick set contains it — for k=4 over
  a 12-layer universe that is a 3x saving on the train split.

``picks`` rows come from ``argsort(rand)[:, :k]`` (the precompute's seeded draw), so positions
within a row are DISTINCT — each (row, layer) pair maps to exactly one output slot, which is
what makes the scatter below collision-free. The ``written == b*k`` assert is the cheap proof.
"""

from typing import Any

import torch
from torch import Tensor


def arout_micro_batch(
    recon: Any,
    input_ids: Tensor,
    attention_mask: Tensor,
    *,
    picks: "Tensor | None",
    keep_pos: list[int],
    no_group: bool = False,
) -> Tensor:
    """AR predictions for one collated micro-batch -> ``[b, n_store, d]`` fp32 on device.

    ``picks``: ``[b, k]`` positions in the AO layer universe (k-sliced train split) or ``None``
    (store every universe layer — the eval split). ``keep_pos``: universe position -> row in the
    AR's OWN layer list (identity unless ``--layer-min/max`` trims the AR's list).
    ``no_group=True`` forces the full-sweep-then-gather path for prompt_tag — the debug arm the
    smoke uses to prove the grouped path bitwise-equal on real weights.
    """
    is_ptag = hasattr(recon, "tag_ids")
    if not is_ptag or no_group or picks is None:
        preds: Tensor
        if is_ptag:
            # full sweep: one tagged forward per AR layer -> [b, n_ar_layers, d]
            preds = recon(input_ids=input_ids, attention_mask=attention_mask, layer_idx=None)
        else:
            preds = recon(input_ids=input_ids, attention_mask=attention_mask)
        if picks is not None:
            idx = picks.long().to(preds.device)
            # picks index the AO universe; map through keep_pos into the AR's row space
            kp = torch.as_tensor(keep_pos, device=preds.device, dtype=torch.long)
            idx = kp[idx]
            return preds.gather(1, idx.unsqueeze(-1).expand(-1, -1, preds.shape[-1]))
        if len(keep_pos) != preds.shape[1]:
            return preds[:, keep_pos, :]
        return preds

    # grouped prompt_tag path: k forwards per crop, batched by layer position
    b, k = picks.shape
    d = recon.head.linear.weight.shape[0]
    out = torch.zeros(b, k, d, dtype=torch.float32, device=input_ids.device)
    picks_dev = picks.long().to(input_ids.device)
    if k > 1:
        sorted_p = picks_dev.sort(dim=1).values
        assert bool((sorted_p[:, 1:] != sorted_p[:, :-1]).all()), (
            "duplicate layer pick within a row — picks must be an argsort-based draw "
            "(distinct positions per row); a repeated layer silently halves that crop's coverage"
        )
    written = 0
    for li in range(len(keep_pos)):
        sel = (picks_dev == li).nonzero()  # [(row, slot)] — at most one slot per row
        if sel.numel() == 0:
            continue
        rows, slots = sel[:, 0], sel[:, 1]
        pred = recon(
            input_ids=input_ids[rows],
            attention_mask=attention_mask[rows],
            layer_idx=keep_pos[li],
        )  # [n_sel, 1, d]
        out[rows, slots] = pred[:, 0].to(out.dtype)
        written += int(rows.shape[0])
    assert written == b * k, (
        f"grouped prompt_tag arout wrote {written} of {b * k} slots — duplicate or "
        "out-of-universe picks (draw must be argsort-based, distinct per row)"
    )
    return out
