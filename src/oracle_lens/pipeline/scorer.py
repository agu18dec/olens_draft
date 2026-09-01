"""Frozen-AR scorer: text -> reconstructed activation -> whitened-MSE score.

Used by (a) the A4 cross-eval against the previous reconstructor's frozen eval set, (b) the
round-0 real-vs-shuffled gate, and (c) the Miles custom reward (`--custom-rm-path`), which must
reproduce exactly this scoring. Two invariants are load-bearing:

- **Reward-path tokenization is BARE** (user decision): the AR trains on bare span ids with no
  chat template, so scored text is stripped of template/special markers and tokenized with
  ``add_special_tokens=False``. A template wrap here would silently depress every score.
- **Checkpoint keys carry ``_orig_mod.``** (saved from compile-wrapped blocks): the loader must
  wrap blocks in ``torch.compile`` identically BEFORE ``PeftModel.from_pretrained`` or the
  adapter silently matches nothing (~2%-retrieval failure mode, oracle-lens runbook).

Aggregation modes over a per-activation text list (the compositional seam):
``single`` scores texts[0]; ``sum`` scores the SUM of per-text reconstructions (the
``sum . map ar . av = id`` certificate design); ``nnls`` refits non-negative coefficients
(``oracle_lens.nnomp.nnls_refit``).
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import torch
from jaxtyping import Bool, Float
from torch import Tensor

from oracle_lens.core.nnomp import nnls_refit
from oracle_lens.core.whitening import Whitener
from oracle_lens.readout_text import strip_scaffolding

AggMode = Literal["single", "sum", "nnls"]


def prepare_reward_text(text: str) -> str:
    """Strip template/special markers; the result is what gets bare-tokenized."""
    return strip_scaffolding(text)


def bare_token_ids(tokenizer: Any, text: str, *, max_len: int = 1024) -> list[int]:
    """Bare tokenization for the reward path: no chat template, no special tokens."""
    cleaned = prepare_reward_text(text)
    ids: list[int] = tokenizer(cleaned, add_special_tokens=False)["input_ids"]
    special = set(tokenizer.all_special_ids)
    ids = [i for i in ids if i not in special]
    return ids[:max_len]


@dataclass
class FrozenReconstructor:
    """Frozen truncated-27B + LoRA + head; eval mode, no grads, ever."""

    backbone: Any  # PeftModel over the truncated decoder (compile-wrapped blocks)
    head: Any  # ReconstructorHead
    tokenizer: Any
    whitener: Whitener
    device: str = "cuda"
    micro_batch: int = 64

    def reconstruct(self, texts: list[str]) -> tuple[Float[Tensor, "n d"], Bool[Tensor, "n"]]:
        """Text -> raw-space reconstruction; rows that strip to empty are flagged invalid."""
        from oracle_lens.core.dump import left_pad_ids

        id_rows = [bare_token_ids(self.tokenizer, t) for t in texts]
        valid = torch.tensor([len(r) > 0 for r in id_rows], dtype=torch.bool)
        d_model = int(self.head.linear.in_features)
        out = torch.zeros(len(texts), d_model, dtype=torch.float32)
        keep = [i for i, r in enumerate(id_rows) if r]
        # bucket by length so padded widths are stable-ish; simple 64-row micro-batches
        keep.sort(key=lambda i: len(id_rows[i]))
        with torch.no_grad():
            for lo in range(0, len(keep), self.micro_batch):
                sel = keep[lo : lo + self.micro_batch]
                width = max(len(id_rows[i]) for i in sel)
                packed = torch.full((len(sel), width), -1, dtype=torch.long)
                lens = torch.zeros(len(sel), dtype=torch.long)
                for row, i in enumerate(sel):
                    packed[row, : len(id_rows[i])] = torch.tensor(id_rows[i], dtype=torch.long)
                    lens[row] = len(id_rows[i])
                ids, attn = left_pad_ids(packed, lens, pad_id=0, device=self.device, width=width)
                hidden = self.backbone(
                    input_ids=ids, attention_mask=attn, use_cache=False
                ).last_hidden_state[:, -1, :]
                pred = self.head(hidden).float().cpu()
                for row, i in enumerate(sel):
                    out[i] = pred[row]
        return out, valid

    def score(
        self,
        texts_per_act: list[list[str]],
        targets: Float[Tensor, "m d"],
        *,
        mode: AggMode = "single",
    ) -> dict[str, Float[Tensor, "m"]]:
        """Whitened-MSE (the reward is its negation) + whitened cosine per activation.

        Rows whose texts all strip to empty get ``valid=False`` and NaN scores — the caller
        (e.g. the Miles RM) decides the floor.
        """
        flat = [t for texts in texts_per_act for t in texts]
        counts = [len(texts) for texts in texts_per_act]
        recon_flat, valid_flat = self.reconstruct(flat)
        m = len(texts_per_act)
        d = recon_flat.shape[1] if flat else int(self.head.linear.in_features)
        pred = torch.zeros(m, d, dtype=torch.float32)
        valid = torch.zeros(m, dtype=torch.bool)
        pos = 0
        max_k = max(counts, default=1)
        vec_buf = torch.zeros(m, max_k, d, dtype=torch.float32)
        vec_valid = torch.zeros(m, max_k, dtype=torch.bool)
        for row, k in enumerate(counts):
            vecs = recon_flat[pos : pos + k]
            ok = valid_flat[pos : pos + k]
            pos += k
            valid[row] = bool(ok.any())
            if mode == "single":
                first = int(ok.nonzero()[0]) if ok.any() else 0
                pred[row] = vecs[first]
            elif mode == "sum":
                pred[row] = vecs[ok].sum(dim=0) if ok.any() else vecs.sum(dim=0)
            else:  # nnls: refit below, batched
                vec_buf[row, :k] = vecs
                vec_valid[row, :k] = ok
        targets = targets.float()
        if mode == "nnls":
            coeffs, _fve = nnls_refit(vec_buf, targets, vec_valid)
            pred = torch.bmm(coeffs.unsqueeze(1), vec_buf).squeeze(1)
        pred_w = self.whitener.whiten(pred)
        target_w = self.whitener.whiten(targets)
        mse = ((pred_w - target_w) ** 2).mean(dim=-1)
        cos = torch.nn.functional.cosine_similarity(pred_w, target_w, dim=-1)
        # THE reward metric (user decision 2026-07-19, matches AR training + NLA -mse_nrm):
        # unit-normalized whitened MSE = 2(1-cos); reward = -this, range [-4, 0]
        unit_mse = 2.0 * (1.0 - cos)
        nan = torch.full_like(mse, float("nan"))
        return {
            "unit_whitened_mse": torch.where(valid, unit_mse, nan),
            "whitened_mse": torch.where(valid, mse, nan),
            "whitened_cos": torch.where(valid, cos, nan),
            "valid": valid.float(),
        }


def load_frozen_reconstructor(
    ckpt_dir: Path,
    whitening_path: Path,
    *,
    model_id: str,
    layer: int,
    ridge_c: float,
    device: str = "cuda",
) -> FrozenReconstructor:
    """Load a saved run (lora/ + head.pt) with the compile-key convention. GPU-side only."""
    from peft import PeftModel
    from transformers import AutoTokenizer

    from oracle_lens.core.reconstructor import ReconstructorHead, truncate_backbone
    from oracle_lens.core.whitening import load_whitener
    from oracle_lens.model import load_causal_lm

    model = load_causal_lm(model_id, dtype=torch.bfloat16, device=device)
    inner = truncate_backbone(model, layer=layer)
    # compile-wrap blocks so `_orig_mod.` LoRA keys match (see module docstring)
    for i in range(len(inner.layers)):
        inner.layers[i] = torch.compile(inner.layers[i], dynamic=False)
    backbone = PeftModel.from_pretrained(inner, str(ckpt_dir / "lora"))
    backbone.eval()
    config = model.config
    d_model = int(getattr(config, "text_config", config).hidden_size)
    state = torch.load(ckpt_dir / "head.pt", map_location=device, weights_only=True)
    # LN is non-affine (no norm.* keys) — read the run config, never key-sniff
    import json as _json

    run_json = ckpt_dir.parent.parent / "runs" / f"{ckpt_dir.name}.json"
    layer_norm = True
    if run_json.exists():
        layer_norm = bool(_json.loads(run_json.read_text())["config"].get("head_layer_norm", True))
    head = ReconstructorHead(d_model, layer_norm=layer_norm)
    head.load_state_dict(state)
    head = head.to(device)
    head.eval()
    for p in backbone.parameters():
        p.requires_grad_(False)
    whitener = load_whitener(whitening_path, ridge_c=ridge_c, device="cpu")
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    return FrozenReconstructor(
        backbone=backbone, head=head, tokenizer=tokenizer, whitener=whitener, device=device
    )
