"""Reward math for the AO GRPO runs — the typed core behind ``scripts/ola/rl/ao_reward.py``.

The reward for a rollout text ``z`` with gold activation ``h`` at layer ``l`` is

    r(h, z) = -2 · (1 - cos( W_l(AR(z)[l] - μ_l), W_l(h - μ_l) ))        ∈ [-4, 0]

— the unit-normalized whitened MSE, identical to the AR's training loss and to
``scripts/ola/miles_reward.py`` (single-layer ancestor). FVE = cos²_w is returned beside it
(the eval metric of ``scripts/ola/ao_fve_eval.py``). No shaping, no format penalty; a rollout
whose ``<explanation>`` extraction is empty scores ``FAILED_EXTRACTION_REWARD``.

The space is configurable via :class:`RewardSpace`:
- ``whiten=False``: score on centered raw vectors (``x - μ_l``) instead of whitened ones —
  the "raw ruler" of ``ao_fve_eval``.
- ``unit_norm=False``: ``r = -MSE`` of the (whitened) vectors instead of the cosine form —
  magnitude-sensitive, the ``whitened_mse`` twin from ``ola.scorer``.

Text → AR-input convention mirrors ``ao_fve_eval`` phase 2 exactly: strip scaffolding
(``prepare_reward_text``), bare-tokenize (no chat template, no specials), truncate to the AR's
64-token span width, right-pad via ``ml_collate`` (the LC AR reads the last REAL token through
the attention mask).
"""

import re
from collections.abc import Callable
from dataclasses import dataclass
from typing import cast

import torch
from jaxtyping import Float, Int
from torch import Tensor
from transformers import PreTrainedTokenizerBase

from oracle_lens.core.whitening import Whitener
from oracle_lens.pipeline.multilayer import LAYERS
from oracle_lens.pipeline.scorer import bare_token_ids, prepare_reward_text

FAILED_EXTRACTION_REWARD = -4.0  # the floor of -2(1-cos) at cos = -1
AR_SPAN_WIDTH = 64  # the AR is only defined on <=64-token spans (its training crops)

_EXPLANATION_RE = re.compile(r"<explanation>(.*?)(</explanation>|$)", re.DOTALL)


@dataclass(frozen=True)
class RewardSpace:
    """Which space the reward is computed in. Default = the design of record."""

    whiten: bool = True  # whitened (per-layer whitener) vs centered-raw
    unit_norm: bool = True  # cosine form -2(1-cos) vs magnitude-sensitive -MSE


DEFAULT_SPACE = RewardSpace()


def extract_explanation(text: str) -> str:
    """``<explanation>…</explanation>`` body (or the whole text), scaffolding-stripped."""
    m = _EXPLANATION_RE.search(text or "")
    return prepare_reward_text(m.group(1) if m else (text or ""))


def reward_text_ids(
    tokenizer: PreTrainedTokenizerBase, text: str, *, max_span: int = AR_SPAN_WIDTH
) -> list[int]:
    """Bare token ids for the AR, truncated to its span width. Empty = failed extraction."""
    ids: list[int] = bare_token_ids(tokenizer, text)
    return ids[:max_span]


@dataclass
class RewardResult:
    """Per-row outputs. ``valid=False`` rows carry the floor reward and NaN diagnostics."""

    reward: Float[Tensor, " n"]
    fve: Float[Tensor, " n"]  # cos^2 in the scored space
    cos: Float[Tensor, " n"]
    mse: Float[Tensor, " n"]  # mean squared error in the scored space
    valid: Tensor  # bool [n]


def score_rows(
    preds: Float[Tensor, "n d"],
    golds: Float[Tensor, "n d"],
    layers: Int[Tensor, " n"],
    whiteners: dict[int, Whitener],
    space: RewardSpace = DEFAULT_SPACE,
) -> RewardResult:
    """Score AR predictions against gold activations, each row in its OWN layer's space."""
    n, _ = preds.shape
    cos = torch.full((n,), float("nan"))
    mse = torch.full((n,), float("nan"))
    for ly in sorted({int(x) for x in layers.tolist()}):
        rows = torch.nonzero(layers == ly).squeeze(-1)
        w = whiteners[ly]
        if space.whiten:
            p = w.whiten(preds[rows])
            g = w.whiten(golds[rows])
        else:
            p = preds[rows].float() - w.mu.to(preds.device)
            g = golds[rows].float() - w.mu.to(golds.device)
        cos[rows] = torch.nn.functional.cosine_similarity(p, g, dim=-1).cpu()
        mse[rows] = ((p - g) ** 2).mean(dim=-1).cpu()
    reward = -2.0 * (1.0 - cos) if space.unit_norm else -mse
    valid = ~torch.isnan(reward)
    reward = torch.where(valid, reward, torch.full_like(reward, FAILED_EXTRACTION_REWARD))
    return RewardResult(reward=reward, fve=cos**2, cos=cos, mse=mse, valid=valid)


def ar_positions(n_emb_rows: int) -> dict[int, int]:
    """layer -> row index in the AR's output, derived from its layer-embedding row count
    (the iolens FINAL has 16 rows — layer 0 dropped; the row count is authoritative)."""
    ar_layers = tuple(LAYERS[-n_emb_rows:]) if n_emb_rows != len(LAYERS) else tuple(LAYERS)
    return {ly: i for i, ly in enumerate(ar_layers)}


def capture_block_outputs(
    backbone: torch.nn.Module,
    layers: list[int],
    input_ids: Tensor,
    attention_mask: Tensor,
) -> dict[int, Tensor]:
    """Residual stream AFTER each requested block, via forward hooks — [b, seq, d] per layer.

    Semantically identical to ``output_hidden_states``'s ``hs[layer + 1]`` but independent
    of the modeling code implementing that kwarg (transformers 5.3's qwen3_5 forward does
    NOT — observed IndexError in the miles env, job 47496). Equivalence is unit-tested
    against ``output_hidden_states`` in the project venv (test_rl_reward).
    """
    blocks = None
    for module in backbone.modules():
        cand = getattr(module, "layers", None)
        if isinstance(cand, torch.nn.ModuleList):
            blocks = cand
            break
    assert blocks is not None, "no decoder ModuleList named 'layers' found"
    captured: dict[int, Tensor] = {}
    handles = []

    def make_hook(ly: int) -> Callable[[torch.nn.Module, object, object], None]:
        def hook(_m: torch.nn.Module, _i: object, out: object) -> None:
            h = out[0] if isinstance(out, tuple) else out
            assert isinstance(h, Tensor)
            captured[ly] = h.detach()

        return hook

    for ly in layers:
        handles.append(blocks[ly].register_forward_hook(make_hook(ly)))
    try:
        with torch.no_grad():
            backbone(input_ids=input_ids, attention_mask=attention_mask, use_cache=False)
    finally:
        for h in handles:
            h.remove()
    missing = [ly for ly in layers if ly not in captured]
    assert not missing, f"blocks {missing} produced no output — truncated backbone?"
    return captured


def score_texts(
    recon: torch.nn.Module,  # LayerConditionedReconstructor (forward(ids, mask) -> [b, L, d])
    tokenizer: PreTrainedTokenizerBase,
    texts: list[str],
    golds: Float[Tensor, "n d"],
    layers: Int[Tensor, " n"],
    whiteners: dict[int, Whitener],
    *,
    space: RewardSpace = DEFAULT_SPACE,
    micro_batch: int = 32,
    pad_id: int = 0,
    device: str = "cuda",
) -> RewardResult:
    """The full text → AR → reward path (extraction assumed already done by the caller).

    Empty texts never reach the AR: they get the floor directly. Non-empty rows are
    length-sorted into micro-batches (right-padded to each batch's max via ``ml_collate``).
    """
    from oracle_lens.pipeline.multilayer_reconstructor import ml_collate

    n = len(texts)
    assert golds.shape[0] == n and layers.shape[0] == n
    id_rows: list[list[int]] = [reward_text_ids(tokenizer, t) for t in texts]
    keep = [i for i, ids in enumerate(id_rows) if ids]

    # layer -> layer_emb-row map: the recon carries its OWN layer list (set by the class
    # and by load_lc_reconstructor); fall back to the 27B LAYERS-suffix rule otherwise.
    rec_layers = getattr(recon, "layers", None)
    if rec_layers is not None:
        pos = {int(ly): i for i, ly in enumerate(rec_layers)}
    else:
        pos = ar_positions(int(recon.layer_emb.weight.shape[0]))  # type: ignore[union-attr, index]
    preds = torch.zeros(n, golds.shape[-1])
    if keep:
        order = sorted(keep, key=lambda i: len(id_rows[i]))
        emb_w = recon.layer_emb.weight  # type: ignore[union-attr]
        with torch.no_grad():
            for lo in range(0, len(order), micro_batch):
                sel = order[lo : lo + micro_batch]
                batch = ml_collate(
                    [{"ids": torch.tensor(id_rows[i]), "target": torch.zeros(1)} for i in sel],
                    pad_id=pad_id,
                )
                ids_dev = batch["input_ids"].to(device)
                mask_dev = batch["attention_mask"].to(device)
                need = sorted({int(layers[i]) for i in sel})
                # hook-based capture (NOT recon.forward/output_hidden_states — see
                # capture_block_outputs); then the LC head math verbatim:
                # pred = head(h_layer[last_real_token] + layer_emb[row])
                cap = capture_block_outputs(
                    recon.backbone,  # type: ignore[arg-type]
                    need,
                    ids_dev,
                    mask_dev,
                )
                last = mask_dev.sum(dim=1) - 1
                for row, i in enumerate(sel):
                    ly = int(layers[i])
                    h_l = cap[ly][row, int(last[row])].float()
                    pred = recon.head(h_l + emb_w[pos[ly]])  # type: ignore[operator, index]
                    preds[i] = pred.float().cpu()

    result = score_rows(preds, golds.float().cpu(), layers, whiteners, space)
    # rows that never reached the AR are failed extractions, not zero-vector predictions
    for i in range(n):
        if i not in keep:
            result.reward[i] = FAILED_EXTRACTION_REWARD
            result.fve[i] = float("nan")
            result.cos[i] = float("nan")
            result.mse[i] = float("nan")
            result.valid[i] = False
    return result


def split_bullets(text: str, k_max: int) -> list[str]:
    """Parse a rollout into <= k_max `- ` bullet concepts (the concepts_raw output).

    Mirrors ao_assemble_distill's target format: lines beginning `- `. Scaffolding
    (chat/think/explanation tags) is stripped first. If no bullet lines are present,
    fall back to the whole prepared text as a single phrase, so the joint reward
    degrades gracefully to a single readout rather than scoring nothing.
    """
    t = prepare_reward_text(text)
    bullets = [
        ln.strip()[2:].strip()
        for ln in t.splitlines()
        if ln.strip().startswith("- ") and len(ln.strip()) > 2
    ]
    bullets = [b for b in bullets if b][:k_max]
    if bullets:
        return bullets
    whole = t.strip()
    return [whole] if whole else []


def score_texts_joint(
    recon: torch.nn.Module,
    tokenizer: PreTrainedTokenizerBase,
    texts: list[str],
    golds: Float[Tensor, "n d"],
    layers: Int[Tensor, " n"],
    whiteners: dict[int, Whitener],
    *,
    space: RewardSpace = DEFAULT_SPACE,
    k_max: int = 4,
    loo: bool = False,
    loo_lambda: float = 0.0,
    contrastive: float = 0.0,
    n_distractors: int = 8,
    distractor_bank: dict[int, Tensor] | None = None,
    diversity_lambda: float = 0.0,
    micro_batch: int = 32,
    pad_id: int = 0,
    device: str = "cuda",
) -> RewardResult:
    """Per-bullet JOINT reward: reconstruct each rollout's bullets through the AR, then
    one non-negative least-squares refit of the (whitened) gold activation onto the
    bullet images -> a single joint FVE per rollout, which IS the reward.

    This is the metric the distillation optimized (nnls_refit over the row's own <= k_max
    bullets). ``space.whiten`` toggles the whitened basis (default on, the canonical
    yardstick); ``space.unit_norm`` reconstructs from unit concept DIRECTIONS (default on)
    vs raw-magnitude bullet images. A rollout with no parseable text scores FVE 0.0 (it
    explains none of the activation) and valid=False.
    """
    from oracle_lens.core.nnomp import nnls_refit

    n = len(texts)
    d = int(golds.shape[-1])
    per_row = [split_bullets(t, k_max) for t in texts]

    # Flatten (row, phrase) so every bullet is AR-mapped once, batched by length.
    flat: list[tuple[int, str]] = [(i, ph) for i, bl in enumerate(per_row) for ph in bl]
    phrase_vec: list[Tensor | None] = [None] * len(flat)
    layer_list = cast("Tensor", recon.layers)
    pos = {int(ly): j for j, ly in enumerate(layer_list)}
    emb_w = cast("Tensor", recon.layer_emb.weight)  # type: ignore[union-attr]

    id_rows = [reward_text_ids(tokenizer, ph) for _, ph in flat]
    keep = [j for j, ids in enumerate(id_rows) if ids]
    if keep:
        from oracle_lens.pipeline.multilayer_reconstructor import ml_collate

        order = sorted(keep, key=lambda j: len(id_rows[j]))
        with torch.no_grad():
            for lo in range(0, len(order), micro_batch):
                sel = order[lo : lo + micro_batch]
                batch = ml_collate(
                    [{"ids": torch.tensor(id_rows[j]), "target": torch.zeros(1)} for j in sel],
                    pad_id=pad_id,
                )
                ids_dev = batch["input_ids"].to(device)
                mask_dev = batch["attention_mask"].to(device)
                need = sorted({int(layers[flat[j][0]]) for j in sel})
                cap = capture_block_outputs(recon.backbone, need, ids_dev, mask_dev)  # type: ignore[arg-type]
                last = mask_dev.sum(dim=1) - 1
                for row, j in enumerate(sel):
                    ly = int(layers[flat[j][0]])
                    h_l = cap[ly][row, int(last[row])].float()
                    phrase_vec[j] = recon.head(h_l + emb_w[pos[ly]]).float().cpu()  # type: ignore[operator]

    row_js: dict[int, list[int]] = {}
    for j, (i, _) in enumerate(flat):
        row_js.setdefault(i, []).append(j)

    # Assemble [n, k_max, d] basis + valid mask + whitened gold, in the scored space.
    vecs = torch.zeros(n, k_max, d)
    valid = torch.zeros(n, k_max, dtype=torch.bool)
    golds_s = torch.zeros(n, d)
    g = golds.float().cpu()
    for i in range(n):
        w = whiteners[int(layers[i])]
        gi = w.whiten(g[i : i + 1])[0] if space.whiten else (g[i] - w.mu)
        golds_s[i] = gi
        kk = 0
        for j in row_js.get(i, []):
            pv = phrase_vec[j]
            if pv is None:
                continue
            pv = w.whiten(pv.unsqueeze(0))[0] if space.whiten else (pv - w.mu)
            if space.unit_norm:
                pv = pv / (pv.norm() + 1e-9)
            vecs[i, kk] = pv
            valid[i, kk] = True
            kk += 1

    _, fve_full = nnls_refit(vecs, golds_s, valid)  # [n] joint FVE over all bullets
    has = valid.any(dim=1)
    fve_full = torch.where(has, fve_full, torch.zeros_like(fve_full))

    # bullet diversity: mean pairwise (1 - cos) over a row's bullet images (unit dirs
    # if space.unit_norm). ~0 => duplicate bullets; ~1 => orthogonal. Logged via .cos.
    # diversity (pairwise, logged via .cos) and det_div = det(Gram) = squared VOLUME the
    # bullet directions span (0 = parallel/redundant, 1 = orthogonal). det_div is the DPP
    # diversity BONUS: joint FVE keeps the set reconstructive; +lambda*det_div pays for the
    # bullets being mutually distinct, un-gameable by rephrasing (a twin adds 0 volume ->
    # collapses det toward 0, so filler is penalized). Single bullet -> det 1 (no spread to
    # reward, but joint FVE provides the count incentive).
    diversity = torch.zeros(n)
    det_div = torch.zeros(n)
    for i in range(n):
        idx = valid[i].nonzero(as_tuple=True)[0]
        if len(idx) == 0:
            continue
        u = vecs[i, idx]
        u = u / (u.norm(dim=1, keepdim=True) + 1e-9)
        if len(idx) == 1:
            det_div[i] = 1.0
            continue
        cs = (u @ u.T)
        m = ~torch.eye(len(idx), dtype=torch.bool)
        diversity[i] = float((1.0 - cs[m]).mean())
        det_div[i] = float(torch.det(cs + 1e-6 * torch.eye(len(idx))).clamp_min(0.0))

    # ALWAYS compute the LOO sum too (cheap: k extra small refits), so both the
    # plain joint FVE and the LOO reward are logged in every run regardless of
    # which one is the training target. Leave-one-out: sum_j (FVE(all) -
    # FVE(all \ j)) -- duplicate bullets contribute ~0 (a twin still covers them),
    # complementary bullets each pull weight.
    loo_reward = torch.zeros(n)
    for j in range(k_max):
        drop = valid.clone()
        drop[:, j] = False  # leave bullet j out (no-op where slot j was already invalid)
        _, fve_drop = nnls_refit(vecs, golds_s, drop)
        fve_drop = torch.where(drop.any(dim=1), fve_drop, torch.zeros_like(fve_drop))
        marg = (fve_full - fve_drop).clamp_min(0.0)
        loo_reward = loo_reward + torch.where(valid[:, j], marg, torch.zeros_like(marg))

    # CONTRASTIVE: how well THIS row's bullets reconstruct OTHER items' activations at
    # the same layer (same layer -> matching whitener, so the reconstruction is fair).
    # A high distractor FVE means the bullets are generic (they explain many activations,
    # not just this one) -- the signature of source-continuation. The reward is the margin
    # fve_own - beta*max_distractor, so generic readouts are penalized and the policy is
    # pushed toward DISTINCTIVE, identifying descriptions.
    distractor_fve = torch.zeros(n)
    if contrastive > 0.0:
        lay = [int(x) for x in (layers.tolist() if torch.is_tensor(layers) else layers)]
        n_dist = int(n_distractors)
        if distractor_bank is not None:
            # distractors from a FIXED per-layer gold bank (batch-independent — the RL
            # batch has too few prompts/layer to draw from). Each bank vector is already
            # whitened in the scored space; refit THIS row's bullets onto n_dist sampled
            # same-layer bank activations and keep the best fit.
            for lyr in set(lay):
                rows = [i for i in range(n) if lay[i] == lyr]
                bank = distractor_bank.get(lyr)
                if bank is None or len(bank) == 0 or not rows:
                    continue
                stride = max(1, len(bank) // n_dist)
                sel = list(range(0, len(bank), stride))[:n_dist]
                rr = torch.tensor(rows)
                vsub, valsub = vecs[rr], valid[rr]
                best = torch.full((len(rows),), -1.0)
                for si in sel:
                    gd = bank[si].unsqueeze(0).expand(len(rows), d)
                    _, fc = nnls_refit(vsub, gd, valsub)
                    best = torch.maximum(best, fc)
                distractor_fve[rr] = best.clamp_min(0.0)
        else:
            # fallback: same-layer distractors within the current batch (needs a batch
            # with several prompts per layer to be meaningful).
            by_layer: dict[int, list[int]] = {}
            for i in range(n):
                by_layer.setdefault(lay[i], []).append(i)
            dist_idx = torch.arange(n).unsqueeze(1).repeat(1, n_dist)  # default: self (masked out)
            dist_mask = torch.zeros(n, n_dist, dtype=torch.bool)
            for _lay, idxs in by_layer.items():
                if len(idxs) < 2:
                    continue
                for i in idxs:
                    others = [j for j in idxs if j != i]
                    stride = max(1, len(others) // n_dist)
                    sel = others[::stride][:n_dist]
                    for c, j in enumerate(sel):
                        dist_idx[i, c] = j
                        dist_mask[i, c] = True
            cross = torch.zeros(n, n_dist)
            for c in range(n_dist):
                _, fve_c = nnls_refit(vecs, golds_s[dist_idx[:, c]], valid)
                cross[:, c] = fve_c
            cross = torch.where(dist_mask, cross, torch.full_like(cross, -1.0))
            distractor_fve = cross.max(dim=1).values.clamp_min(0.0)
        distractor_fve = torch.where(has, distractor_fve, torch.zeros_like(distractor_fve))

    if contrastive > 0.0:     # margin: explain THIS activation better than any distractor
        reward = fve_full - contrastive * distractor_fve
        reward = torch.where(has, reward, torch.full_like(reward, FAILED_EXTRACTION_REWARD))
    elif loo_lambda > 0.0:    # joint + λ·LOO: joint keeps the 4-bullet format & FVE,
        reward = fve_full + loo_lambda * loo_reward   # λ·LOO adds anti-filler pressure
    elif loo:
        reward = loo_reward
    else:
        reward = fve_full
    if diversity_lambda > 0.0:  # joint + lambda*det(Gram): reconstructive AND diverse bullets
        reward = reward + diversity_lambda * det_div
    # .fve = plain joint FVE (always); .mse slot carries the LOO reward (or the distractor
    # FVE under contrastive) for logging the OTHER signal; .cos carries bullet diversity.
    aux = distractor_fve if contrastive > 0.0 else loo_reward
    return RewardResult(reward=reward.clone(), fve=fve_full.clone(),
                        cos=diversity.clone(), mse=aux.clone(),
                        valid=has.clone())
