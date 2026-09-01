"""Refit per-layer whiteners on on-policy data (the v2 metric/loss yardstick).

The whitener defines the *training loss*, not just the eval metric, so both cells of the v2
ladder must train under the SAME whitener. User decision 2026-07-25: refit on-policy. Started
before the pt-on pairs existed, so the fit data is **chat-on train rows only** — the
deployment-relevant cell; pt-on later trains under this same artifact, keeping cells comparable.

"Properly" constraints implemented here:
- fit rows exclude exactly the rows ``train_ml`` assigns to its eval split (same filters, same
  ``n_eval`` head-carve, same pool ordering), so the metric is never fit on rows it will judge;
- moments accumulate in fp64 (mean + covariance per layer) over mmap'd targets, chunked;
- provenance (shards, row counts, git commit) is embedded in each safetensors file's metadata
  and written beside the artifacts as ``provenance.json``.

Output: ``<out_dir>/whitening_L{L}.safetensors`` — drop-in for ``load_whitener`` (ridge at load).
"""

import json
from pathlib import Path
from typing import Any

from oracle_lens.pipeline.jobs.settings import MODEL_ID
from oracle_lens.pipeline.paths import git_commit


def refit_whiteners(
    *,
    root: Path,
    pairs_dir: str,
    out_dir: Path,
    n_eval_reserved: int = 2000,
    max_rows: int = 400_000,
    chunk: int = 2048,
    device: str = "cuda",
) -> dict[str, Any]:
    """Accumulate per-layer μ/Σ over train-split target rows and write whitening_L*.safetensors."""
    import torch
    from transformers import AutoTokenizer

    from oracle_lens.core.whitening import save_moments
    from oracle_lens.pipeline.multilayer import load_multilayer_shards_lazy

    paths = sorted(
        p
        for d in pairs_dir.split(",")
        for p in (root / d.strip()).glob("pairs_train_*.safetensors")
    )
    if not paths:
        raise FileNotFoundError(f"no pairs_train_*.safetensors under {root} / {pairs_dir}")
    pairs, meta = load_multilayer_shards_lazy(paths)
    layers = list(meta.layers)

    # mirror train_ml's filter + split EXACTLY, so the reserved eval head never touches the fit
    _im = AutoTokenizer.from_pretrained(MODEL_ID).convert_tokens_to_ids("<|im_start|>")
    assert isinstance(_im, int)
    hits = torch.nonzero(pairs.span_ids == _im).squeeze(-1)
    keep = torch.ones(len(pairs), dtype=torch.bool)
    if len(hits):
        bad = torch.unique(torch.searchsorted(pairs.offsets, hits, right=True) - 1)
        keep[bad] = False
    lens = pairs.lengths
    keep &= (lens >= 1) & (lens <= 512)
    pairs = pairs.select(torch.nonzero(keep).squeeze(-1))
    n = len(pairs)
    n_reserved = min(n_eval_reserved, n // 10)
    train_idx = torch.arange(n_reserved, n)  # everything past the eval head
    if len(train_idx) > max_rows:
        # evenly-spaced subsample keeps shard coverage uniform (rows are shard-ordered)
        sel = torch.linspace(0, len(train_idx) - 1, max_rows).round().long()
        train_idx = train_idx[sel]
    print(f"[refit] pool={n} rows, reserved eval head={n_reserved}, fitting on {len(train_idx)}",
          flush=True)

    d = 5120
    nl = len(layers)
    s1 = torch.zeros(nl, d, dtype=torch.float64, device=device)
    s2 = torch.zeros(nl, d, d, dtype=torch.float64, device=device)
    count = 0
    for lo in range(0, len(train_idx), chunk):
        idx = train_idx[lo : lo + chunk]
        x = torch.stack([pairs.targets[int(i)] for i in idx]).to(device).to(torch.float64)
        s1 += x.sum(dim=0)  # [nl, d]
        s2 += torch.einsum("bld,ble->lde", x, x)
        count += len(idx)
        if lo % (chunk * 20) == 0:
            print(f"  {lo}/{len(train_idx)}", flush=True)

    out_dir.mkdir(parents=True, exist_ok=True)
    prov = {
        "model_id": MODEL_ID,
        "pairs_dir": pairs_dir,
        "shards": [p.name for p in paths],
        "rows_fit": count,
        "n_eval_reserved": int(n_reserved),
        "layers": layers,
        "git_commit": git_commit(),
        "note": "chat-on train rows only (pt-on pairs not yet available at fit time); "
        "both v2 cells train and eval under this artifact",
    }
    commit = str(prov["git_commit"])
    for li, layer in enumerate(layers):
        mu = (s1[li] / count).float()
        cov = (s2[li] / count - torch.outer(s1[li] / count, s1[li] / count)).float()
        save_moments(
            out_dir / f"whitening_L{layer}.safetensors",
            mu.cpu(),
            cov.cpu(),
            meta={"rows": str(count), "source": "v2-refit-chat-on", "git_commit": commit},
        )
    (out_dir / "provenance.json").write_text(json.dumps(prov, indent=2))
    print(f"[refit] wrote {nl} whiteners -> {out_dir}", flush=True)
    return {"rows_fit": count, "layers": layers, "out_dir": str(out_dir)}
