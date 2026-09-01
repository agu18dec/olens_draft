"""Fresh multi-layer reconstructor: text -> the residual at prev_pos for layers {20,24,..,60}.

Design (user): a SINGLE full-backbone forward, read the FINAL-layer hidden at the phrase's last
real token, then a per-layer head maps it to that layer's activation. No torch.compile (fresh
model, no LoRA-key-matching reason to compile), right-pad + gather the last token (no left-pad),
per-layer whitened-cosine loss. This is the plain multi-layer analogue of the single-layer AR
(``oracle_lens.reconstructor``); it trains from a base model (not warm-started from ola-ar-v2).
"""

import json
import os
from collections.abc import Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import torch
from jaxtyping import Float, Int
from mytorch_lightning.config import TrainingConfig
from mytorch_lightning.entry import do_train
from mytorch_lightning.entry import train as mtl_train
from mytorch_lightning.mydule import Mydule
from torch import Tensor, nn
from torch.utils.data import Dataset, DistributedSampler

from oracle_lens.core.reconstructor import ReconstructorHead
from oracle_lens.core.whitening import Whitener
from oracle_lens.pipeline.jspace import MetricSpace
from oracle_lens.pipeline.multilayer import (
    LAYERS,
    CroppedPairs,
    MultiLayerPairs,
    StreamingCropDataset,
    TokenBudgetSampler,
    token_budget_nbatches,
)

PairsLike = MultiLayerPairs | CroppedPairs


class MultiLayerReconstructor(nn.Module):
    """Full backbone (decoder + LoRA) + one head per target layer, read off the final hidden."""

    def __init__(self, backbone: Any, heads: nn.ModuleList) -> None:
        super().__init__()
        self.backbone = backbone
        self.heads = heads

    def forward(
        self, input_ids: Int[Tensor, "b p"], attention_mask: Int[Tensor, "b p"]
    ) -> Float[Tensor, "b n_layers d"]:
        out = self.backbone(input_ids=input_ids, attention_mask=attention_mask, use_cache=False)
        # right-padded: the last REAL token is at index (mask.sum(1) - 1), gather it per row
        last = attention_mask.sum(dim=1) - 1
        final = out.last_hidden_state[torch.arange(input_ids.shape[0]), last]  # [b, d]
        preds = torch.stack([head(final) for head in self.heads], dim=1)  # [b, n_layers, d]
        return preds


class LayerConditionedReconstructor(nn.Module):
    """Full backbone (LoRA) + ONE shared head conditioned on a per-layer embedding, reading EACH
    target layer's OWN residual (matched-depth). ``preds[:, li] = head(h_{layer} + emb(li))`` — a
    single Linear head for all layers, the layer identity injected via a learned embedding.
    Param-efficient (1 head, not n) and can interpolate to HELD-OUT layers (train on a subset of
    layers, evaluate on the rest). The AR analogue of the layer-conditioned AO.
    """

    def __init__(
        self, backbone: Any, layers: tuple[int, ...], d_model: int, *, layer_norm: bool = True
    ) -> None:
        super().__init__()
        self.backbone = backbone
        self.layers = list(layers)
        self.layer_emb = nn.Embedding(len(self.layers), d_model)
        nn.init.normal_(self.layer_emb.weight, std=0.02)
        self.head = ReconstructorHead(d_model, layer_norm=layer_norm)  # fp32, shared across layers

    def forward(
        self, input_ids: Int[Tensor, "b p"], attention_mask: Int[Tensor, "b p"]
    ) -> Float[Tensor, "b n_layers d"]:
        out = self.backbone(
            input_ids=input_ids,
            attention_mask=attention_mask,
            use_cache=False,
            output_hidden_states=True,
        )
        hs = out.hidden_states  # tuple: hs[0]=embeddings, hs[i]=output of block i-1
        rows = torch.arange(input_ids.shape[0], device=input_ids.device)
        last = attention_mask.sum(dim=1) - 1  # right-pad: last REAL token per row
        emb = self.layer_emb.weight  # [n_layers, d] (fp32)
        preds = []
        for li, lyr in enumerate(self.layers):
            # residual AT layer `lyr` (output of block lyr) = hs[lyr+1]; read the span's last token
            h_l = hs[lyr + 1][rows, last].float()  # [b, d]
            preds.append(self.head(h_l + emb[li]))
        return torch.stack(preds, dim=1)  # [b, n_layers, d]


class PromptTagReconstructor(nn.Module):
    """Layer identity in the PROMPT, not the architecture.

    Input is ``[Layer 44] <span>``; the read is the FINAL hidden state at the span's last real
    token; ONE shared :class:`ReconstructorHead` emits the prediction. There is **no layer
    embedding and no per-layer head** — the only thing telling the model which layer to
    reconstruct is text. Contrast :class:`LayerConditionedReconstructor`, which reads each layer's
    OWN residual and injects the layer through a learned embedding into the head (a strictly
    linear-in-that-residual readout).

    ``tag_ids`` is a frozen ``[n_layers, T_tag]`` buffer of pre-tokenized tags, all of the SAME
    length, so a batch keeps one static shape and ``torch.compile(dynamic=False)`` still compiles
    one graph per block. The tag is prepended, so the collate's right-padding stays at the tail and
    ``attention_mask.sum(1) - 1`` still finds the last real token.

    Two call shapes:
      * ``layer_idx=li`` -> ``[b, 1, d]``: ONE forward, the training path.
      * ``layer_idx=None`` -> ``[b, n_layers, d]``: n_layers forwards, the validation path, so
        every val metric is form-identical to the other architecture's.
    """

    def __init__(
        self,
        backbone: Any,
        layers: tuple[int, ...],
        d_model: int,
        tag_ids: Int[Tensor, "n_layers t"],
        *,
        layer_norm: bool = True,
    ) -> None:
        super().__init__()
        if tag_ids.shape[0] != len(layers):
            raise ValueError(f"tag_ids has {tag_ids.shape[0]} rows for {len(layers)} layers")
        self.backbone = backbone
        self.layers = list(layers)
        self.register_buffer("tag_ids", tag_ids.long(), persistent=True)
        self.head = ReconstructorHead(d_model, layer_norm=layer_norm)  # fp32, shared

    @property
    def tags(self) -> Int[Tensor, "n_layers t"]:
        """The tag buffer, typed. ``register_buffer`` widens to ``Tensor | Module`` for mypy."""
        t = self.tag_ids
        assert isinstance(t, Tensor)
        return t

    def _tagged(
        self, input_ids: Int[Tensor, "b p"], attention_mask: Int[Tensor, "b p"], layer_idx: int
    ) -> tuple[Tensor, Tensor]:
        tag = self.tags[layer_idx].to(input_ids.device).expand(input_ids.shape[0], -1)
        return (
            torch.cat([tag, input_ids], dim=1),
            torch.cat([torch.ones_like(tag), attention_mask], dim=1),
        )

    def _read_final(self, input_ids: Tensor, attention_mask: Tensor) -> Tensor:
        out = self.backbone(input_ids=input_ids, attention_mask=attention_mask, use_cache=False)
        rows = torch.arange(input_ids.shape[0], device=input_ids.device)
        last = attention_mask.sum(dim=1) - 1
        h: Tensor = out.last_hidden_state[rows, last]
        return h.float()

    def forward(
        self,
        input_ids: Int[Tensor, "b p"],
        attention_mask: Int[Tensor, "b p"],
        layer_idx: int | None = None,
    ) -> Float[Tensor, "b n_sel d"]:
        if layer_idx is not None:
            ids, mask = self._tagged(input_ids, attention_mask, layer_idx)
            one: Tensor = self.head(self._read_final(ids, mask))
            return one.unsqueeze(1)
        preds: list[Tensor] = [
            self.head(self._read_final(*self._tagged(input_ids, attention_mask, li)))
            for li in range(self.tags.shape[0])
        ]
        return torch.stack(preds, dim=1)


def multilayer_whitened_cosine_loss(
    preds: Float[Tensor, "b n_layers d"],
    targets: Float[Tensor, "b n_layers d"],
    whiteners: Sequence[MetricSpace | None],
) -> Float[Tensor, ""]:
    """Mean over layers and batch of ``2(1 - cos)`` in each layer's own metric space.

    A ``None`` entry EXCLUDES that layer from the loss (e.g. L63 under the J-space ruler, which
    has no Jacobian) — the mean runs over the remaining layers only, never a silent n-as-N mean.
    """
    losses = []
    for li, w in enumerate(whiteners):
        if w is None:
            continue
        pw = w.whiten(preds[:, li])
        tw = w.whiten(targets[:, li].float())
        uh = pw / pw.norm(dim=-1, keepdim=True).clamp_min(1e-9)
        u = tw / tw.norm(dim=-1, keepdim=True).clamp_min(1e-9)
        losses.append(((uh - u) ** 2).sum(dim=-1).mean())
    if not losses:
        raise ValueError("all layers excluded from the loss (every metric space is None)")
    return torch.stack(losses).mean()


def multilayer_recon_loss(
    preds: Float[Tensor, "b n_layers d"],
    targets: Float[Tensor, "b n_layers d"],
    whiteners: Sequence[MetricSpace | None],
    *,
    mode: str = "whiten",
    loss_spaces: Sequence[MetricSpace | None] | None = None,
) -> Float[Tensor, ""]:
    """Training-loss dispatcher — the iolens AR-loss ablation knob.

    ``whiten`` (config of record): per-layer whitened cosine, delegates unchanged.
    ``jspace`` / ``mixed``: the SAME ``2(1 - cos)`` form through ``loss_spaces`` — per-layer
    ``JSpace`` rulers (pure J) or ``MixedSpace`` rulers whose cosine is exactly the λ-blend of
    the whitened and J cosines (see ola.jspace). The ruler list encodes the space; a ``None``
    slot (L63, no Jacobian) excludes that layer loudly.
    ``rawcos``: the same ``2(1 - cos)`` form on RAW activations — no whitener anywhere in the
    loss. (A cosine against a unit-normed target would be identical: cosine normalizes both
    sides, which is why the third arm must be MSE.)
    ``unitnorm``: MSE to the UNIT-NORMED raw GT activation — trains the head to emit an
    approximately unit vector, distinguishable from ``rawcos`` by the norm penalty.

    Val metrics (FVE/retrieval/val_loss) stay in the whitened basis for every arm, so the arms
    are comparable by construction (J-bearing arms additionally log val_jfve_*/val_jloss).
    """
    if mode == "whiten":
        return multilayer_whitened_cosine_loss(preds, targets, whiteners)
    if mode in ("jspace", "mixed"):
        if loss_spaces is None:
            raise ValueError(f"loss_space={mode!r} needs loss_spaces (J/Mixed rulers)")
        return multilayer_whitened_cosine_loss(preds, targets, loss_spaces)
    losses = []
    for li in range(preds.shape[1]):
        p = preds[:, li]
        t = targets[:, li].float()
        u = t / t.norm(dim=-1, keepdim=True).clamp_min(1e-9)
        if mode == "rawcos":
            uh = p / p.norm(dim=-1, keepdim=True).clamp_min(1e-9)
            losses.append(((uh - u) ** 2).sum(dim=-1).mean())
        elif mode == "unitnorm":
            losses.append(((p - u) ** 2).sum(dim=-1).mean())
        else:
            raise ValueError(
                f"unknown loss_space {mode!r} (whiten | rawcos | unitnorm | jspace | mixed)"
            )
    return torch.stack(losses).mean()


def head_state(model: Any) -> dict[str, Any]:
    """Serialize whichever head family this AR has, so a new arch cannot crash the save path.

    Dispatch is on the ATTRIBUTES present, not on a head_mode string, and every branch is
    self-describing on load: ``layer_emb`` => layer_conditioned, ``tag_ids`` => prompt_tag, a bare
    ``heads`` ModuleList => read_final. Previously this unconditionally read ``layer_emb`` inside a
    ``hasattr(m, "head")`` branch, so any single-head arch without an embedding raised
    AttributeError at the FIRST milestone — after the training was already paid for.
    """
    if hasattr(model, "head"):
        state: dict[str, Any] = {"head": model.head.state_dict()}
        if hasattr(model, "layer_emb"):
            state["layer_emb"] = model.layer_emb.state_dict()
        if hasattr(model, "tag_ids"):
            state["tag_ids"] = model.tags.detach().cpu()
            state["layers"] = list(getattr(model, "layers", []))
        return state
    return {"heads": model.heads.state_dict()}


def multilayer_fve(
    preds: Float[Tensor, "b n_layers d"],
    targets: Float[Tensor, "b n_layers d"],
    whiteners: Sequence[MetricSpace],
) -> Float[Tensor, "n_layers"]:
    """Per-layer FVE = mean cos^2 (whitened) — the fitted-coefficient reconstruction quality."""
    out = []
    for li, w in enumerate(whiteners):
        pw = w.whiten(preds[:, li])
        tw = w.whiten(targets[:, li].float())
        cos = torch.nn.functional.cosine_similarity(pw, tw, dim=-1)
        out.append((cos**2).mean())
    return torch.stack(out)


def multilayer_retrieval_top1(
    preds: Float[Tensor, "b n_layers d"],
    targets: Float[Tensor, "b n_layers d"],
    whiteners: Sequence[MetricSpace],
    *,
    pool: int = 100,
    seed: int = 0,
) -> Float[Tensor, "n_layers"]:
    """Per-layer top-1 retrieval: is the TRUE target the pred's nearest (whitened-cos) neighbour
    among a pool of ``pool`` candidates? Rows are shuffled (seeded) then chunked into disjoint
    pools; leftover rows short of a full pool are dropped (chance = 1/pool). Complements FVE:
    FVE can be middling while retrieval is high (pred is closest to ITS OWN target even if the
    absolute fit is loose) — the metric that matters for 'does the AR pin the right activation'."""
    n = preds.shape[0]
    n_pools = n // pool
    if n_pools == 0:
        return torch.zeros(len(whiteners))
    order = torch.randperm(n, generator=torch.Generator().manual_seed(seed))[: n_pools * pool]
    out = []
    for li, w in enumerate(whiteners):
        pw = w.whiten(preds[order, li])
        tw = w.whiten(targets[order, li].float())
        pw = pw / pw.norm(dim=-1, keepdim=True).clamp_min(1e-9)
        tw = tw / tw.norm(dim=-1, keepdim=True).clamp_min(1e-9)
        hits = 0
        for k in range(n_pools):
            sl = slice(k * pool, (k + 1) * pool)
            sim = pw[sl] @ tw[sl].T  # [pool, pool] pred_i x target_j
            hits += int((sim.argmax(dim=1) == torch.arange(pool)).sum())
        out.append(hits / (n_pools * pool))
    return torch.tensor(out)


@dataclass
class MLReconConfig:
    """One multi-layer reconstructor run (fresh from base, no compile)."""

    run_name: str
    n_layers: int = 11
    n_pairs: int = 0  # 0 = all
    match_length_to: str = ""  # if set, subsample train pool to this pairs-dir's length histogram
    min_len: int = 1
    max_len: int = 512
    head_layer_norm: bool = True
    head_mode: str = "read_final"  # "read_final" (per-layer heads) | "layer_conditioned" (shared)
    whitener_prefix: str = "whitening"  # per-layer whitener prefix (whitening_onpolicy = on-policy)
    span_law: str = ""  # ""=pool's own (loguniform) | "uniform"=paper N~uniform{min_len..max_len}
    crop_max: int = 0  # >0: stratified-uniform prefix crops, N exactly uniform in {1..crop_max}
    # save_every_steps>0: periodic resume saves. Two consumers, one knob: jobs/train.py's
    # ResumeCheckpoint callback (NOTE: the streaming sampler has no step-exact data
    # fast-forward, so that path restores weights/optimizer but REPLAYS the data order from the
    # start — jobs/train.py warns loudly), and the iolens Mydule's atomic lightweight resume
    # blob (LoRA+heads+optimizer+state; the harness's own save serialises the whole
    # PEFT-wrapped 27B, ~54 GB).
    save_every_steps: int = 0
    crop_seed: int = 1234  # crop-pool draw seed (view 2 uses a fresh seed + exclusion of view 1)
    crop_exclude_seeds: str = ""  # csv of prior pool seeds: rebuild + EXCLUDE all their pairs
    crop_exclude_per_n: int = 0  # the prior pool's per-N count (needed to reconstruct it exactly)
    init_from: str = ""  # warm-start from ml_checkpoints/<name> (LoRA + heads) — continuation runs
    grad_checkpointing: bool = True  # REQUIRED: 64-layer no-ckpt activations OOM even at 4096 tok
    pad_width: int = 0  # >0: fixed right-pad width (static shapes for compile; crop32 -> 32)
    compile_blocks: bool = False  # per-block torch.compile(dynamic=False); requires pad_width>0
    localize_targets: bool = False  # crop mode: copy rung targets to a local file (space!)
    localize_path: str = "/tmp/targets_local.bin"  # /dev/shm/... for >disk sets (RAM-backed)
    stream_targets: bool = True  # crop mode: storage-order streaming + shuffle buffer (no capacity
    # walls; exact global shuffle when the buffer covers the per-rank stream — rungs <= ~1.2M)
    stream_buffer_rows: int = 150_000  # per-rank shuffle buffer (~26 GB of 1 TiB RAM)
    bucket_by_length: bool = True  # token-budget batches: cut padding -> less recompute -> faster
    token_budget: int = 8192  # max padded tokens (rows * max_len) per micro-batch (memory guard)
    max_batch_rows: int = 8  # = micro_batch: keep eff-batch ~= old runs so FVE stays comparable;
    # the budget only SHRINKS a batch below this for the longest spans (>1024 tok) to bound memory
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    lr: float = 1e-4
    lr_sched: str = "constant"
    ridge_c: float = 0.1
    micro_batch: int = 8
    grad_accum: int = 4
    epochs: int = 1
    eval_every_steps: int = 200
    max_eval_rows: int = 1024
    warmup_steps: int = 20
    pad_id: int = 0
    seed: int = 0
    # loss_ridge_c>0 and != ridge_c: the TRAINING loss whitens with this ridge while all val
    # metrics keep ridge_c — tests whether down-weighting the near-noise direction tail in the
    # objective concentrates gradient on the predictable structure (user hypothesis 2026-07-28).
    # Mutually exclusive with loss_space != "whiten".
    loss_ridge_c: float = 0.0
    # loss_space: the AR-loss ablation arm — see multilayer_recon_loss. Val metrics stay whitened
    # in every arm. "jspace" runs the loss in the Jacobian-lens space z=(x-μ)@Jᵀ on the J-covered
    # layers only (L63 excluded loudly; val additionally logs val_jfve_*/val_jloss); "mixed" is
    # (1-lam)*whitened + lam*J via jspace.MixedSpace, so lam=0/1 are the whitened/pure-J endpoints.
    loss_space: str = "whiten"  # "whiten" | "rawcos" | "unitnorm" | "jspace" | "mixed"
    # J-loss arms (loss_space jspace/mixed): the pinned J-lens artifact the loss rulers are built
    # from, and the whitened/J interpolation weight for "mixed" (0=pure whitened, 1=pure J).
    # Defaults keep old run-config JSONs round-tripping through MLReconConfig(**json.loads(...)).
    loss_mix_lambda: float = 0.5
    jspace_repo: str = ""
    jspace_file: str = ""
    jspace_revision: str = ""
    # ckpt_every_steps>0: milestone checkpoints step{N}/{lora,heads.pt,meta.json} with EXACT
    # all-reduced span-token counts — the checkpoint series IS the AR scaling curve (constant LR).
    ckpt_every_steps: int = 0
    # layer_indices: subset of LAYERS to train on (indices, not layer numbers). The AR drops
    # layer 0 — its target is essentially "which token preceded this span" and scores at chance
    # (measured 0.4% FVE vs 20.4% at L63), so it wastes head capacity and drags the mean.
    layer_indices: tuple[int, ...] = tuple(range(len(LAYERS)))
    # stream_dir: train off a live producer-fed buffer instead of a fixed pool (see stream_pairs)
    stream_dir: str = ""
    max_steps_override: int = 0
    # ckpt_samples_base>0: LOG-SPACED sample-axis milestones instead — save when the exact
    # example count crosses base * factor^k (k=0,1,...). Log-log scaling curves need log-spaced
    # x points; uniform-in-steps saving starves the low end and gaps the high end.
    ckpt_samples_base: int = 0
    ckpt_samples_factor: float = 1.4142135623730951
    # prompt_tag arch only: the layer is named in the PROMPT. tag_ids_json is the pre-tokenized
    # tag per TRAINED layer, resolved once in main() so all ranks share one tokenization and the
    # exact ids land in every milestone meta.json as provenance.
    tag_template: str = "[Layer {n:02d}]"
    tag_width: int = 0
    tag_ids_json: str = ""
    extra: dict[str, str] = field(default_factory=dict)

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


class MLReconDataset(Dataset[dict[str, Tensor]]):
    def __init__(
        self, pairs: MultiLayerPairs | CroppedPairs, layer_indices: tuple[int, ...] | None = None
    ) -> None:
        self.pairs = pairs
        # stored shards carry all of LAYERS; the trained subset (e.g. layer 0 dropped) is sliced
        # here so train and val see the SAME stack as the model and the whiteners
        self.layer_indices = layer_indices

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, i: int) -> dict[str, Tensor]:
        target = self.pairs.targets[i].float()
        if self.layer_indices is not None:
            target = target[list(self.layer_indices)]
        return {"ids": self.pairs.row_ids(i), "target": target}


def ml_collate(rows: list[dict[str, Tensor]], *, pad_id: int, width: int = 0) -> dict[str, Tensor]:
    """Right-pad ids to batch max — or to the FIXED ``width`` (static shape for per-block
    torch.compile; rows must fit). Last real token at ``mask.sum-1``; targets [b, L, d]."""
    maxlen = max(int(r["ids"].shape[0]) for r in rows)
    if width:
        if maxlen > width:
            raise ValueError(f"row of length {maxlen} exceeds fixed pad_width {width}")
        maxlen = width
    b = len(rows)
    ids = torch.full((b, maxlen), pad_id, dtype=torch.long)
    attn = torch.zeros(b, maxlen, dtype=torch.long)
    for i, r in enumerate(rows):
        n = int(r["ids"].shape[0])
        ids[i, :n] = r["ids"]
        attn[i, :n] = 1
    return {
        "input_ids": ids,
        "attention_mask": attn,
        "targets": torch.stack([r["target"] for r in rows]),
    }


class MLReconMydule(Mydule):  # type: ignore[misc]
    """Per-layer whitened-cosine training of the multi-layer reconstructor (no compile)."""

    def __init__(
        self,
        model: MultiLayerReconstructor,
        cfg: MLReconConfig,
        train_pairs: MultiLayerPairs | CroppedPairs,
        eval_pairs: MultiLayerPairs | CroppedPairs,
        whiteners: list[Whitener],
        *,
        loss_whiteners: Sequence[MetricSpace | None] | None = None,
        jspaces: Sequence[MetricSpace | None] | None = None,
        ckpt_dir: Path | None = None,
        skip_batches: int = 0,
        tokens_span_prev: int = 0,
        examples_prev: int = 0,
    ) -> None:
        super().__init__()
        self._model = model
        self._cfg = cfg
        self._whiteners = whiteners  # metric ruler (val loss/FVE/retrieval) — never the loss knob
        self._loss_whiteners: Sequence[MetricSpace | None] = (
            loss_whiteners if loss_whiteners is not None else whiteners
        )
        # PURE-J rulers for val metrics, kept separate from the loss ruler: under loss_space=
        # "mixed" the loss ruler is a MixedSpace whose cosine is a blend, so reading val_jfve off
        # it would report neither space. This way val_jfve_* means the same thing in every arm.
        self._jspaces = jspaces
        li = cfg.layer_indices if len(cfg.layer_indices) != len(LAYERS) else None
        self._train_data = MLReconDataset(train_pairs, li)
        self._eval_data = MLReconDataset(eval_pairs, li)
        self._val_preds: list[Tensor] = []
        self._val_targets: list[Tensor] = []
        self._n_val_batches = max(1, len(self._eval_data) // cfg.micro_batch)
        self.last_val_metrics: dict[str, float] = {}
        # iolens: exact per-rank span-token counter (all-reduced at validation); prev carries a
        # resumed run's already-counted tokens so the milestone x-axis stays exact across resumes.
        self._ckpt_dir = ckpt_dir
        self._skip_batches = skip_batches
        self._tokens_span_rank = 0
        self._tokens_span_prev = tokens_span_prev
        self.tokens_span_total = tokens_span_prev
        self._examples_rank = 0
        self._examples_prev = examples_prev
        self.examples_total = examples_prev
        self._fp_logged = False  # step-0 batch fingerprint (cross-arm data-order audit)
        self._next_milestone = float(cfg.ckpt_samples_base) if cfg.ckpt_samples_base else 0.0
        # Fast-forward the milestone threshold past where this run RESUMED, or every restart
        # writes a rung immediately: examples_total already exceeds ckpt_samples_base, so the
        # first validation saves at the resume point instead of the next geometric step. Five
        # restarts in ten minutes (a supervisor thrashing on false stalls) left five rungs inside
        # 2,300 samples of each other on pt's curve — same x, slightly different y, silently
        # overweighting that point in every fit.
        if self._next_milestone:
            while self._next_milestone <= examples_prev:
                self._next_milestone *= cfg.ckpt_samples_factor

    def create_model(self) -> torch.nn.Module:
        return self._model

    def configure_optimizer(self) -> torch.optim.Optimizer:
        params = [p for p in self._model.parameters() if p.requires_grad]
        # fused=True: the trainable set is hundreds of small LoRA tensors; the unfused optimizer
        # pays a kernel launch per tensor per step. Same math, fused accumulation.
        fused = os.environ.get("OLA_FUSED_ADAMW", "1") == "1" and torch.cuda.is_available()
        return torch.optim.AdamW(params, lr=self._cfg.lr, fused=fused)

    def initialize_model(self, model: torch.nn.Module) -> None:
        # DDP re-broadcasts ALL module buffers every forward (broadcast_buffers=True default,
        # not exposed by the harness) — measured 352 ms/step of nccl:broadcast for buffers that
        # never change (rotary caches etc.). Every rank builds the model identically (same
        # snapshot, same seed, same resume blob), so skipping the broadcast is safe.
        model._ddp_params_and_buffers_to_ignore = [  # type: ignore[assignment]
            name for name, _ in model.named_buffers()
        ]

    def train_data(self) -> MLReconDataset:
        return self._train_data

    def val_data(self) -> MLReconDataset:
        return self._eval_data

    def configure_training_dl(self, args: dict[str, Any]) -> dict[str, Any]:
        pad_id = self._cfg.pad_id
        width = self._cfg.pad_width
        args["collate_fn"] = lambda rows: ml_collate(rows, pad_id=pad_id, width=width)
        # prefetch mmap targets/ids off the main process (LazyTargets reopens handles per worker)
        args["num_workers"] = 4
        args["persistent_workers"] = True
        args["pin_memory"] = True
        data: Any = args.get("dataset")
        is_train = data is self._train_data
        world = self.trainer.config.world_size() if (is_train and self.trainer) else 1
        rank = self.trainer.global_rank if (is_train and self.trainer) else 0
        if is_train and self._cfg.stream_dir:
            # live buffer: shards claimed per rank, rows split across workers, consumed shards
            # marked so the janitor can free them (see ola.stream_pairs)
            from oracle_lens.pipeline.stream_pairs import StreamingPairDataset, stream_stats

            print(
                f"[ml-recon] rank{rank}/{world} stream: "
                f"{stream_stats(Path(self._cfg.stream_dir), world)}",
                flush=True,
            )
            args["dataset"] = StreamingPairDataset(
                Path(self._cfg.stream_dir),
                rank=rank,
                world=world,
                layer_indices=self._cfg.layer_indices,
                seed=self._cfg.seed,
            )
            args["num_workers"] = 4
            args["persistent_workers"] = False
            for k in ("sampler", "shuffle"):
                args.pop(k, None)
            return args
        if is_train and self._cfg.stream_targets and isinstance(data.pairs, CroppedPairs):
            # storage-order streaming + shuffle buffer (see StreamingCropDataset): sequential
            # volume reads, no local materialization. Iterable => sampler/shuffle must go;
            # ONE worker (more would duplicate the stream).
            args["dataset"] = StreamingCropDataset(
                data.pairs,
                rank=rank,
                world=world,
                buffer_rows=self._cfg.stream_buffer_rows,
                seed=self._cfg.seed,
                skip_batches=self._skip_batches,
                micro_batch=self._cfg.micro_batch,
            )
            # 8 worker PROCESSES, each a disjoint sub-stride (get_worker_info) — parallel volume
            # reads; a single reader starves the GPUs (blocking per-row mmap reads)
            args["num_workers"] = 8
            args["persistent_workers"] = False
            for k in ("sampler", "shuffle"):
                args.pop(k, None)
            return args
        if is_train and self._cfg.bucket_by_length and not width:
            lengths = data.pairs.lengths.tolist()
            args["batch_sampler"] = TokenBudgetSampler(
                lengths,
                self._cfg.token_budget,
                world=world,
                rank=rank,
                seed=self._cfg.seed,
                max_rows=self._cfg.max_batch_rows,
            )
            for k in ("batch_size", "shuffle", "sampler", "drop_last"):  # excl. w/ batch_sampler
                args.pop(k, None)
        elif world > 1:
            args["sampler"] = DistributedSampler(
                data, num_replicas=world, rank=rank, shuffle=True, seed=self._cfg.seed
            )
            args.pop("shuffle", None)
        return args

    def configure_validation_dl(self, args: dict[str, Any]) -> dict[str, Any]:
        return self.configure_training_dl(args)

    def _forward(
        self, batch: dict[str, Tensor], layer_idx: int | None = None
    ) -> tuple[Tensor, Tensor]:
        wrapped = getattr(self, "model", None)
        fwd = wrapped if wrapped is not None else self._model
        if layer_idx is None:
            preds = fwd(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"])
            return preds.float(), batch["targets"].float()
        # prompt_tag training path: one layer per micro-batch, so slice the targets to match.
        preds = fwd(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            layer_idx=layer_idx,
        )
        return preds.float(), batch["targets"][:, layer_idx : layer_idx + 1].float()

    def _schedule_layer_idx(self) -> int:
        """Stratified layer schedule for the prompt_tag arch — NOT i.i.d. sampling.

        The layer is a *prompt*, so one micro-batch can only supervise one layer. Assigning
        ``slot = (micro_step % ga) * world + rank`` spreads the ``ga * world`` micro-batches of one
        optimizer step across distinct layers; since the harness divides the loss by ``ga`` and DDP
        averages over ranks, the optimizer-step gradient is then *mean over layers of mean over
        rows* — the same estimator form as the layer-conditioned arch's, which is what makes the
        two curves a fair comparison rather than two different objectives. The ``+ g // ga`` term
        rotates which rank owns which layer across steps so no rank is pinned to one layer.

        ``global_step`` is fast-forwarded by ``RestoreTrainerState``, so the phase survives resume.
        """
        n = len(self._cfg.layer_indices)
        tr = self.trainer
        if tr is None:
            return 0
        ga = max(1, int(tr.config.gradient_accumulation_steps))
        world = max(1, int(tr.config.world_size()))
        g = int(tr.global_step)
        slot = (g % ga) * world + int(tr.global_rank)
        return (slot + g // ga) % n

    def training_step(self, batch: dict[str, Tensor], batch_info: Any) -> Tensor:
        if not self._fp_logged:
            # one line per run: identical pool + seed + world MUST print identical fingerprints
            # across arms — the decisive data-order audit for the fair loss-space comparison
            import hashlib

            fp = hashlib.blake2b(
                batch["input_ids"].cpu().numpy().tobytes(), digest_size=8
            ).hexdigest()
            print(
                f"[ml-recon] step0 batch fingerprint {fp} "
                f"(shape {tuple(batch['input_ids'].shape)})",
                flush=True,
            )
            self._fp_logged = True
        self._tokens_span_rank += int(batch["attention_mask"].sum())
        self._examples_rank += int(batch["attention_mask"].shape[0])
        if self._cfg.head_mode == "prompt_tag":
            li = self._schedule_layer_idx()
            preds, targets = self._forward(batch, layer_idx=li)
            lw = [self._loss_whiteners[li]]
            return multilayer_recon_loss(
                preds, targets, lw, mode=self._cfg.loss_space, loss_spaces=lw
            )
        preds, targets = self._forward(batch)
        return multilayer_recon_loss(
            preds,
            targets,
            self._loss_whiteners,
            mode=self._cfg.loss_space,
            loss_spaces=self._loss_whiteners,
        )

    def _sync_token_totals(self) -> int:
        """Exact global span-token + example counts: all-reduce(SUM) per-rank counters + prev."""
        t = torch.tensor([self._tokens_span_rank, self._examples_rank], dtype=torch.long)
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            t = t.to(torch.cuda.current_device())
            torch.distributed.all_reduce(t, op=torch.distributed.ReduceOp.SUM)
        self.tokens_span_total = self._tokens_span_prev + int(t[0].item())
        self.examples_total = self._examples_prev + int(t[1].item())
        return self.tokens_span_total

    def _save_milestone(self, step: int, metrics: dict[str, float]) -> None:
        """Rank-0 milestone checkpoint ex{N}/{lora, heads.pt, meta.json} — the scaling-curve
        rung. Saved at validations whose optimizer step crossed a ckpt_every_steps multiple.

        Named by CUMULATIVE EXAMPLES, not micro-step: micro-steps restart at 0 on every warm
        restart, so ``step{N}`` collides with a rung from an earlier segment — and the
        exists-guard below turns that collision into a *silently skipped* rung, not an
        overwrite. ``examples_total`` carries ``--examples-prev`` and so is monotone across
        the whole series, which is also exactly the curve's x-axis.
        """
        assert self._ckpt_dir is not None
        out = self._ckpt_dir / f"ex{self.examples_total}"
        tmp = self._ckpt_dir / f"ex{self.examples_total}.tmp"
        if out.exists():
            return
        tmp.mkdir(parents=True, exist_ok=True)
        self._model.backbone.save_pretrained(str(tmp / "lora"))
        torch.save(head_state(self._model), tmp / "heads.pt")
        meta = {
            "micro_steps": step,
            "tokens_span": self.tokens_span_total,
            "examples": self.examples_total,
            **{k: float(v) for k, v in metrics.items()},
            "config": self._cfg.to_dict(),
        }
        (tmp / "meta.json").write_text(json.dumps(meta, indent=2))
        os.replace(tmp, out)
        # name the DIRECTORY, not the micro-step: they differ since milestones went to ex{N},
        # and a log line that says step22205 while the rung on disk is ex2880960 sends anyone
        # grepping for a checkpoint to a path that does not exist
        print(
            f"[ml-recon] milestone {out.name}: examples={self.examples_total:,} "
            f"tokens_span={self.tokens_span_total:,} (micro_step {step})",
            flush=True,
        )

    def _save_resume(self, step: int) -> None:
        """Rank-0 atomic lightweight resume blob (LoRA + heads + optimizer + state)."""
        assert self._ckpt_dir is not None
        from oracle_lens.pipeline.resume import save_resume_state

        opt = self.trainer.optimizer if self.trainer is not None else None
        world = self.trainer.config.world_size() if self.trainer is not None else 1
        save_resume_state(
            self._ckpt_dir,
            recon=self._model,
            peft_inner=self._model.backbone,
            optimizer=opt,
            global_step=step,
            tokens_span=self.tokens_span_total,
            examples=self.examples_total,
            world_size=world,
            num_workers=8,  # the streaming path's fixed worker count (replay depends on it)
        )

    def validation_step(self, batch: dict[str, Tensor], batch_info: Any) -> None:
        with torch.no_grad():
            preds, targets = self._forward(batch)
        self._val_preds.append(preds.detach().cpu())
        self._val_targets.append(targets.detach().cpu())
        # per-rank count under DDP (mytorch shards val) so the val_fve block actually fires
        world = self.trainer.config.world_size() if self.trainer is not None else 1
        n_expected = max(1, (len(self._eval_data) // max(1, world)) // self._cfg.micro_batch)
        if len(self._val_preds) >= n_expected:
            preds_all = torch.cat(self._val_preds)
            targets_all = torch.cat(self._val_targets)
            cpu_w = [w.to("cpu") for w in self._whiteners]
            fve = multilayer_fve(preds_all, targets_all, cpu_w)
            loss = float(multilayer_whitened_cosine_loss(preds_all, targets_all, cpu_w))
            ret1 = multilayer_retrieval_top1(preds_all, targets_all, cpu_w, seed=self._cfg.seed)
            metrics = {
                "val_loss": loss,
                "val_fve_mean": float(fve.mean()),
                "val_ret1_mean": round(float(ret1.mean()), 4),
            }
            # RAW (non-whitened) FVE: identity-W spaces centered by each layer's pooled mean.
            # Same cos^2 estimator, no covariance rotation — the "what fraction of the raw
            # activation direction is recovered" number (user request 2026-08-19).
            from oracle_lens.core.whitening import Whitener as RawSpace

            raw_w = [
                RawSpace(mu=w.mu.cpu(), w=torch.eye(w.mu.shape[0]), ridge_c=0.0) for w in cpu_w
            ]
            rawfve = multilayer_fve(preds_all, targets_all, raw_w)
            metrics["val_rawfve_mean"] = float(rawfve.mean())
            sub = [LAYERS[i] for i in self._cfg.layer_indices]
            for li, f in enumerate(rawfve.tolist()):
                metrics[f"val_rawfve_L{sub[li]}"] = round(f, 4)
            for li, f in enumerate(fve.tolist()):
                metrics[f"val_fve_L{sub[li]}"] = round(f, 4)
            for li, r in enumerate(ret1.tolist()):
                metrics[f"val_ret1_L{sub[li]}"] = round(r, 4)
            if self._jspaces is not None:
                # J-space metrics over the COVERED layers only (None = excluded, e.g. L63);
                # the whitened metrics above keep all trained layers as the shared signal.
                cov = [li for li, s in enumerate(self._jspaces) if s is not None]
                cpu_j = [s.to("cpu") for s in self._jspaces if s is not None]
                jpred, jtgt = preds_all[:, cov], targets_all[:, cov]
                jfve = multilayer_fve(jpred, jtgt, cpu_j)
                jret1 = multilayer_retrieval_top1(jpred, jtgt, cpu_j, seed=self._cfg.seed)
                metrics["val_jloss"] = float(multilayer_whitened_cosine_loss(jpred, jtgt, cpu_j))
                metrics["val_jfve_mean"] = float(jfve.mean())
                metrics["val_jret1_mean"] = round(float(jret1.mean()), 4)
                for ci, f in zip(cov, jfve.tolist(), strict=True):
                    metrics[f"val_jfve_L{sub[ci]}"] = round(f, 4)
                for ci, r in zip(cov, jret1.tolist(), strict=True):
                    metrics[f"val_jret1_L{sub[ci]}"] = round(r, 4)
            if self._cfg.loss_space != "whiten":
                # convergence signal in the TRAINING loss space (metrics above stay whitened)
                cpu_ls: Sequence[MetricSpace | None] | None = None
                if self._cfg.loss_space in ("jspace", "mixed"):
                    cpu_ls = [s.to("cpu") if s is not None else None for s in self._loss_whiteners]
                metrics["val_loss_train_space"] = float(
                    multilayer_recon_loss(
                        preds_all,
                        targets_all,
                        cpu_w,
                        mode=self._cfg.loss_space,
                        loss_spaces=cpu_ls,
                    )
                )
            metrics["tokens_span"] = float(self._sync_token_totals())
            for k, v in metrics.items():
                self.log(k, torch.tensor(float(v)), across_devices=False)
            print(f"[ml-recon] {metrics}", flush=True)
            self.last_val_metrics = metrics
            self._val_preds, self._val_targets = [], []
            # iolens: milestone + resume saves ride the validation cadence (rank 0 only; every
            # rank has just synced tokens_span so the meta is exact).
            rank = self.trainer.global_rank if self.trainer is not None else 0
            step = int(self.trainer.global_step) if self.trainer is not None else 0
            if self._ckpt_dir is not None and rank == 0 and step > 0:
                # The two schedules are ADDITIVE, not exclusive. Log-spaced sample rungs are what
                # make a log-log scaling curve readable (uniform-in-steps crowds the right end and
                # starves the fast-moving left end); a uniform every-N-steps cadence is what makes
                # the run recoverable at a predictable granularity. Both name the dir by cumulative
                # examples and _save_milestone is idempotent on an existing dir, so a step that
                # satisfies both triggers writes exactly one rung.
                due = False
                if self._next_milestone and self.examples_total >= self._next_milestone:
                    # may have crossed several milestones since the last validation — advance past
                    # all of them, save once
                    while self.examples_total >= self._next_milestone:
                        self._next_milestone *= self._cfg.ckpt_samples_factor
                    due = True
                every = self._cfg.ckpt_every_steps
                if every > 0 and step % every < self._cfg.eval_every_steps:
                    due = True
                if due:
                    self._save_milestone(step, metrics)
                if self._cfg.save_every_steps > 0:
                    self._save_resume(step)

    def hyperparams_to_log(self) -> dict[str, object]:
        return self._cfg.to_dict()

    def forward(self, *args: Any, **kwargs: Any) -> Any:
        return self._model(*args, **kwargs)


def build_ml_training_config(
    cfg: MLReconConfig,
    n_train: int,
    *,
    world_size: int = 1,
    wandb_project: str | None = None,
    train_lengths: list[int] | None = None,
) -> TrainingConfig:
    world_size = max(1, world_size)
    grad_accum = max(1, cfg.grad_accum // world_size) if world_size > 1 else cfg.grad_accum
    per_rank = max(1, n_train // world_size)
    tc = TrainingConfig()
    tc.name = cfg.run_name
    tc.max_epochs = cfg.epochs
    if cfg.bucket_by_length and not cfg.pad_width and train_lengths is not None:
        # token-budget schedule: max_steps = per-rank optimizer batches (variable size), not n/mb
        nb = token_budget_nbatches(
            train_lengths, cfg.token_budget, world_size, cfg.max_batch_rows, cfg.seed
        )
        steps = max(1, cfg.epochs * nb // grad_accum)
    else:  # fixed micro_batch (incl. pad_width static mode)
        steps = max(1, cfg.epochs * (per_rank // cfg.micro_batch) // grad_accum)
    tc.max_steps = cfg.max_steps_override or steps
    tc.ngpu = world_size
    tc.train_batch_size = cfg.micro_batch
    tc.val_batch_size = cfg.micro_batch
    tc.gradient_accumulation_steps = grad_accum
    tc.gradient_clip_val = 1.0
    tc.lr = cfg.lr
    tc.lr_sched_type = cfg.lr_sched
    tc.final_lr = 0.0
    tc.warmup_steps = min(cfg.warmup_steps, max(1, steps // 10))
    tc.amp_dtype = None
    tc.validation_step_interval = cfg.eval_every_steps
    tc.validation_epoch_interval = 1
    tc.train_drop_last = True
    tc.val_drop_last = True
    tc.num_workers = 0
    tc.seed = cfg.seed
    tc.base_save_dir = os.environ.get("OLA_CKPT_DIR", "/tmp/mtl-runs")
    # Per-step wandb logging + global grad-norm reduction cost real wall time at short-span step
    # counts; curves at 25-step granularity are equivalent.
    tc.train_log_interval = int(os.environ.get("OLA_LOG_INTERVAL", "25"))
    # A per-job rendezvous port: the harness hardcodes master_port=10210, so two DDP runs on one
    # box deadlock on the rendezvous. env.sh exports a per-job MASTER_PORT under Slurm; honor it
    # here (and let a no-scheduler box set it per run). do_train COPIES tc.master_port back into
    # the env, so exporting MASTER_PORT alone is not enough — it must flow through the config.
    if os.environ.get("MASTER_ADDR"):
        tc.master_addr = os.environ["MASTER_ADDR"]
    if os.environ.get("MASTER_PORT"):
        tc.master_port = int(os.environ["MASTER_PORT"])
    if os.environ.get("OLA_DIST_TIMEOUT"):
        from datetime import timedelta

        tc.dist_timeout = timedelta(seconds=int(os.environ["OLA_DIST_TIMEOUT"]))
    if wandb_project is not None:
        tc.logger = "wandb"
        tc.wandb_config.project = wandb_project
    tc.finalize()
    return tc


def train_ml_reconstructor(
    model: MultiLayerReconstructor,
    cfg: MLReconConfig,
    train_pairs: MultiLayerPairs | CroppedPairs,
    eval_pairs: MultiLayerPairs | CroppedPairs,
    whiteners: list[Whitener],
    *,
    loss_whiteners: Sequence[MetricSpace | None] | None = None,
    jspaces: Sequence[MetricSpace | None] | None = None,
    wandb_project: str | None = None,
    local_rank: int = 0,
    world_size: int = 1,
    callbacks: list[Any] | None = None,
    ckpt_dir: Path | None = None,
    skip_batches: int = 0,
    tokens_span_prev: int = 0,
    examples_prev: int = 0,
) -> dict[str, float]:
    if cfg.max_eval_rows and len(eval_pairs) > cfg.max_eval_rows:
        eval_pairs = eval_pairs.select(torch.arange(cfg.max_eval_rows))
    mydule = MLReconMydule(
        model,
        cfg,
        train_pairs,
        eval_pairs,
        whiteners,
        loss_whiteners=loss_whiteners,
        jspaces=jspaces,
        ckpt_dir=ckpt_dir,
        skip_batches=skip_batches,
        tokens_span_prev=tokens_span_prev,
        examples_prev=examples_prev,
    )
    tc = build_ml_training_config(
        cfg,
        len(mydule._train_data),
        world_size=world_size,
        wandb_project=wandb_project,
        train_lengths=train_pairs.lengths.tolist(),
    )
    if callbacks:
        # do_train does not forward callbacks; replicate its env setup and build the Trainer
        # directly (vendor untouched). Same for the single-GPU path.
        from mytorch_lightning.trainer import Trainer

        os.environ["MASTER_ADDR"] = tc.master_addr
        os.environ["MASTER_PORT"] = str(tc.master_port)
        os.environ["RANK"] = str(tc.local_to_global_rank(local_rank))
        os.environ["LOCAL_RANK"] = str(local_rank)
        os.environ["WORLD_SIZE"] = str(tc.world_size())
        trainer = Trainer(config=tc, local_rank=local_rank, callbacks=callbacks)
        trainer.train(mydule)
    elif world_size > 1:
        do_train(tc, local_rank, mydule)
    else:
        mtl_train(tc, mydule)
    summary = dict(mydule.last_val_metrics)
    summary["tokens_span_total"] = float(mydule.tokens_span_total)
    summary["examples_total"] = float(mydule.examples_total)
    return summary


def build_ml_heads(d_model: int, n_layers: int, *, layer_norm: bool) -> nn.ModuleList:
    """One fp32 LN+Linear head per target layer."""
    heads = [ReconstructorHead(d_model, layer_norm=layer_norm) for _ in range(n_layers)]
    return nn.ModuleList(heads)
