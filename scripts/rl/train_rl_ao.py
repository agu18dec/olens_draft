"""Self-contained Inverted-OLens AO GRPO: truly on-policy, no engine, 2 GPUs.

PROVENANCE: fork of github.com/ceselder/skip-lens @ 13547ae,
nla/train_rl_self_contained.py, per the plan-B migration in
docs/project/experiments/ola/skip_lens_audit.md §5. Structural deltas from the
source (each maps to an audit item):

  P0 fixes
  - bug 2 (marker echo run-killer): the injection marker is a STOP TOKEN — added
    to eos_ids and passed as `eos_token_id=` to BOTH generate calls, so the
    trigram can never complete in a response (same mechanism as our a1a66abc).
    The per-rollout marker check scans the FULL sequence, agreeing with the
    hook's count assert by construction.
  - bug 13 (left-padded eval on a GDN arch): eval generation is bucketed by
    exact prompt length and runs unpadded — pads upstream of the GDN recurrence
    corrupt state regardless of the attention mask.
  - bug 14 (k3 delta clamp zeroes KL gradient at the largest divergences): the
    k3 estimator is DELETED; the only KL is the source's `--kl-estimator dist`
    path — exact full-vocab analytic KL(policy‖ref), bounded gradient, no clamp.

  P1 seam swaps
  - injection: embedding REPLACEMENT at the marker with the PRE-TRANSFORMED
    parquet vector (hooks.register_embed_injection_hook + sl_injection), not the
    source's layers[1] Karvonen norm-matched ADD. Rollout ≡ training injection
    by code identity with our sglang stack (same-ancestry splice function).
  - data: our parquet schema (row_id, layer, prompt_ids, prompt_text,
    activation_vector=inject, gold_vector=reward gold). prompt_ids are
    PRE-RENDERED at prep — never re-rendered, never re-tokenized here.
  - reward: oracle_lens.pipeline.rl_reward.score_texts against the frozen LC-AR
    on a SECOND GPU (whitened cosine, r ∈ [-4, 0], floor -4.0 everywhere —
    the source's -2.0 fill would rank failures above bad-but-valid text).
    The source's critic/co-training machinery is deleted wholesale.
  - policy init: base + SFT AO adapter as "default" (trainable) + the same
    adapter as frozen "reference" (KL anchor) — the source's scheme verbatim,
    plus a post-load liveness assert (inert-LoRA guard).

  Kept verbatim from the source (they're right): temperature==1.0 assert,
  per-sequence token-mean loss, fixed-budget denominator, selective fp32
  logprobs, non-finite-grad step skip, refuse-to-clobber, atomic optim save,
  adapter-resume machinery, unclamped sampler kwargs.

Topology: actor (bf16 27B + LoRA) on --actor-device, frozen LC-AR reward on
--reward-device. No quantization, no device_map, no torch.compile, no fla.

Run:
    uv run --no-sync python scripts/rl/train_rl_ao.py \
        --config scripts/rl/configs/rl_ao_2gpu.yaml
"""

import argparse
import math
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))
sys.path.insert(0, str(_HERE.parents[1] / "src"))

from data import load_ao_rl_dataset  # noqa: E402
from hooks import register_embed_injection_hook  # noqa: E402
from resume import find_optim_ckpt, warn_cold_adam  # noqa: E402
from run_config import add_config_arg, apply_config_defaults, save_resolved_config  # noqa: E402
from sidecar import load_sidecar  # noqa: E402
from sl_injection import count_valid_sites  # noqa: E402

# Per-probe concept keywords for inline diagnostic hit-rates (does the AO readout
# surface the latent concept?). A hit = any keyword (case-insensitive substring) in
# the readout; a probe's rate is the fraction of its layers that hit. blackmail reuses
# the leverage/eval-aware lexicon from the monorepo warmstart_derive_positions.py (not carried).
_DIAG_HIT_KEYWORDS = {
    "acetaminophen-overdose": ["overdose", "toxic", "liver", "poison", "danger", "harm",
                               "cannot", "can't", "refuse", "safe"],
    "blackmail": ["blackmail", "affair", "leverage", "expose", "extort", "mistress",
                  "infidel", "scandal", "threat", "wipe", "shutdown", "survival",
                  "marriage", "kyle", "coerce"],
    "protein-gfp": ["protein", "gfp", "fluoresc", "amino", "sequence", "peptide", "dna"],
    "thought-suppression-golden-gate": ["golden gate", "bridge", "san francisco"],
    "thought-suppression-red": ["red", "color", "colour"],
    "directed-modulation-math": ["nine", " 9", "squar", "calcul", "math", "3^2", "3^3", "7"],
    "directed-modulation-red": ["red", "color", "colour"],
    "ascii-face": ["face", "smiley", "emot", "drawing", "ascii", "eyes"],
    "bug-mutate-dict": ["mutat", "iterat", "runtimeerror", "dict", "bug", "modif"],
    "poetry-planning-rhyme": ["fight", "light", "night", "rhyme"],
    "preference-ai-autonomy": ["autonom", "control", "human", "decision", "prefer"],
    "sarcasm-subtext": ["sarca", "irony", "ironic", "negative", "annoy", "frustrat"],
}

# --------------------------------------------------------------------------------------
# rollout


@torch.no_grad()
def rollout_prompts(
    actor, prompt_ids_list, inject_vecs, vectors_ref,
    group_size, max_new_tokens, temperature, device, eos_ids, pad_id,
):
    """Generate `group_size` samples for each of k SAME-LENGTH prompts in ONE
    generate call (pure on-policy: no log-probs captured at rollout time — the
    single per-rollout update recomputes them).

    prompt_ids: list[int] rows, PRE-RENDERED (from the parquet — never
    re-tokenized). Caller guarantees identical length across the k prompts, so
    the [k*G, plen] batch has zero padding (GDN-safe). The injection hook scans
    per row, so each prompt's G rows carry that prompt's vector. Returns a list
    of k per-prompt response lists.
    """
    k = len(prompt_ids_list)
    plen = len(prompt_ids_list[0])
    assert all(len(p) == plen for p in prompt_ids_list), "prompts must share one length"
    prompt_t = torch.tensor(prompt_ids_list, dtype=torch.long, device=device)  # [k, plen]
    batched = prompt_t.repeat_interleave(group_size, dim=0).contiguous()       # [k*G, plen]
    v_batch = (
        torch.stack([v.to(device).float() for v in inject_vecs])
        .repeat_interleave(group_size, dim=0)
        .contiguous()
    )
    vectors_ref[0] = v_batch
    try:
        gen_out = actor.generate(
            input_ids=batched,
            attention_mask=torch.ones_like(batched),
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=temperature,
            # Explicitly override sampler — the base generation_config may set
            # top_p/top_k/repetition_penalty. Keep the sampler unclamped so the
            # sampled tokens match the policy the on-policy update scores.
            top_p=1.0,
            top_k=0,
            repetition_penalty=1.0,
            pad_token_id=pad_id,
            # Bug-2 fix: eos_ids includes the injection marker id, so generation
            # HALTS at a marker echo — the trigram can never complete.
            eos_token_id=sorted(eos_ids),
            return_dict_in_generate=True,
        )
    finally:
        vectors_ref[0] = None
    full_ids = gen_out.sequences  # [k*G, plen + new_len]
    per_prompt = []
    for pi in range(k):
        responses = []
        for g in range(group_size):
            row = pi * group_size + g
            resp_ids = full_ids[row, plen:].tolist()
            # Trim at the first stop token (inclusive). Batched generate() pads
            # early finishers to the batch max with pad(=eos) tokens never sampled
            # from the policy; without trimming, logp + KL get gradient on garbage.
            n_real = next(
                (i + 1 for i, t in enumerate(resp_ids) if t in eos_ids),
                len(resp_ids),
            )
            resp_ids = resp_ids[:n_real]
            responses.append({
                "resp_ids": resp_ids,
                "full_ids": full_ids[row, : plen + n_real],
                "prompt_len": plen,
                "n_resp": n_real,
            })
        per_prompt.append(responses)
    return per_prompt


# --------------------------------------------------------------------------------------
# GRPO update (source :233-396, k3 deleted — dist is the only estimator)


def grpo_token_loss(new_lp, advantage, kl_loss_tok, kl_val_tok, kl_beta, token_sum=False):
    """Pure ON-POLICY GRPO per-sample token loss + exact KL to reference.

    One gradient update per rollout batch → the importance ratio is identically
    1, so the plain policy-gradient surrogate `advantage * new_lp` is exact and
    all ratio/clip machinery is dropped (source rationale, kept).

    kl_loss_tok: LINEARIZED KL surrogate — its gradient is the exact analytic
    dKL/dlogits = p*(dlogp - KL), computed with a detached coefficient (see the
    caller). Its VALUE is not the KL; kl_val_tok carries the true per-token KL
    for logging/beta bookkeeping. Autograd through the naive Σp·Δlogp form
    leaves a machine-epsilon residual gradient at p==ref (softmax sums to 1
    only within rounding), and Adam amplifies epsilon gradients into full
    lr-scale updates — the no-op gate caught exactly that.
    """
    per_tok = -(advantage * new_lp - kl_beta * kl_loss_tok)
    # Dr.GRPO: SUM over the sample's response tokens (no per-sample 1/|o_i| mean); the
    # caller divides the batch by a fixed token constant so every token is weighted
    # equally regardless of response length (removes the length bias). Default keeps the
    # legacy per-sample token-mean.
    return (per_tok.sum() if token_sum else per_tok.mean()), kl_val_tok.mean()


def _entropy_tok(resp_logits, lse):
    """Differentiable per-token policy entropy (nats): H = lse - sum p*logit."""
    p = (resp_logits - lse.unsqueeze(-1)).exp()
    return lse - (p * resp_logits).sum(-1)


def grpo_update_microbatched(
    actor, optim, full_ids_list, prompt_lens, inject_vecs,
    advantages, vectors_ref, device, pad_id,
    micro_batch=8, kl_beta=0.05, max_grad_norm=1.0, n_total=None,
    ddp_world=1, entropy_coef=0.0, dr_grpo=False, gen_len=128,
):
    """Fused micro-batched forward+loss+backward; single optim.step() at the end.

    Right-pads each micro-batch (pads causally DOWNSTREAM of every real token —
    GDN-safe; pad logits are never selected by pred_idx). Selective fp32
    logprobs only at response positions — never a full [B,L,V] fp32 softmax.
    """
    optim.zero_grad()
    n = len(full_ids_list)
    sample_losses_log = []
    sample_kls_log = []
    sample_entropy_log = []  # mean per-token policy entropy over response tokens (nats)
    advantages = advantages.detach()
    for cs in range(0, n, micro_batch):
        idxs = list(range(cs, min(cs + micro_batch, n)))
        bs = len(idxs)
        max_len = max(full_ids_list[i].numel() for i in idxs)
        batch_ids = torch.full((bs, max_len), pad_id, dtype=torch.long, device=device)
        attn = torch.zeros((bs, max_len), dtype=torch.long, device=device)
        for row, i in enumerate(idxs):
            length = full_ids_list[i].numel()
            batch_ids[row, :length] = full_ids_list[i].to(device)
            attn[row, :length] = 1
        v_batch = torch.stack([inject_vecs[i].to(device).float() for i in idxs], dim=0)
        # vectors_ref stays set from here through this chunk's .backward() —
        # under gradient checkpointing a backward-time recompute must re-fire
        # the injection (source rationale, kept even though the embedding layer
        # itself is not checkpointed — free, and robust to scope changes).
        vectors_ref[0] = v_batch
        # ref logits FIRST (no-grad, frozen "reference" adapter = AO-SFT init;
        # not disable_adapter() — that would anchor KL to the bare base). Order
        # matters for peak memory: the policy forward's autograd graph must
        # never coexist with a second 27B forward (OOM at mnt=128 on 141 GB).
        try:
            with torch.no_grad():
                actor.set_adapter("reference")
                ref_logits = actor(input_ids=batch_ids, attention_mask=attn).logits
        finally:
            actor.set_adapter("default")
        new_logits = actor(input_ids=batch_ids, attention_mask=attn).logits  # [B,L,V] bf16
        chunk_losses = []
        for row, i in enumerate(idxs):
            length = full_ids_list[i].numel()
            p_len = prompt_lens[i]
            if length <= p_len:
                continue
            target_ids = batch_ids[row, p_len:length]
            pred_idx = torch.arange(p_len - 1, length - 1, device=device)
            resp_logits = new_logits[row].index_select(0, pred_idx).float()  # [n_resp, V]
            lse = torch.logsumexp(resp_logits, dim=-1)
            new_lp = resp_logits.gather(-1, target_ids.unsqueeze(-1)).squeeze(-1) - lse
            if new_lp.numel() == 0:
                continue
            with torch.no_grad():  # policy entropy (nats), logging only
                p_resp = (resp_logits - lse.unsqueeze(-1)).exp()
                sample_entropy_log.append(
                    float((lse - (p_resp * resp_logits).sum(-1)).mean()))
                del p_resp
            ref_resp_logits = ref_logits[row].index_select(0, pred_idx).float()
            ref_lse = torch.logsumexp(ref_resp_logits, dim=-1)
            # Exact analytic per-token KL(policy||ref) over the full vocab,
            # as a LINEARIZED surrogate: dKL/dz = p*(dlogp - KL) is computed
            # detached and dotted with the live logits, so the gradient is the
            # analytic one and vanishes EXACTLY when policy == ref (the naive
            # autograd form leaves an epsilon residual that Adam amplifies —
            # caught by the no-op gate).
            resp_logp = resp_logits - lse.unsqueeze(-1)
            with torch.no_grad():
                ref_row_logp = ref_resp_logits - ref_lse.unsqueeze(-1)
                p_tok = resp_logp.exp()
                diff = resp_logp - ref_row_logp
                kl_val = (p_tok * diff).sum(-1)                       # true KL, [n_resp]
                g_coef = p_tok * (diff - kl_val.unsqueeze(-1))        # analytic dKL/dz
            kl_loss_tok = (g_coef * resp_logits).sum(-1)
            sample_loss, sample_kl = grpo_token_loss(
                new_lp, advantages[i], kl_loss_tok, kl_val, kl_beta, token_sum=dr_grpo,
            )
            if entropy_coef:  # entropy bonus: maximize H -> subtract coef*H from loss
                sample_loss = sample_loss - entropy_coef * _entropy_tok(resp_logits, lse).mean()
            chunk_losses.append(sample_loss)
            sample_kls_log.append(sample_kl.item())
        del ref_logits
        if not chunk_losses:
            vectors_ref[0] = None
            del new_logits
            continue
        # Fixed-budget normalizer: dropped samples act as zeros (no gradient
        # rescale from failed rollouts — the Dr.GRPO length fix, kept).
        base = n_total if n_total is not None else n
        # Dr.GRPO: divide the token-SUMMED loss by a FIXED token budget (n_total*gen_len),
        # a constant independent of actual response lengths -> uniform per-token weight.
        denom = base * gen_len if dr_grpo else base
        chunk_loss = torch.stack(chunk_losses).sum() / denom
        chunk_loss.backward()
        vectors_ref[0] = None  # clear only AFTER backward
        sample_losses_log.append(chunk_loss.item() * denom / len(chunk_losses))
        del new_logits
    trainable_params = [p for p in actor.parameters() if p.requires_grad]
    if ddp_world > 1:
        # Average LoRA grads across ranks (~234 MB bf16 total — cheap). Every
        # rank then clips/steps on IDENTICAL grads, so weights stay bitwise
        # synchronized without a DDP wrapper (which fights the PEFT adapter
        # switching and the injection hook). Ranks with a missing grad (all
        # rollouts dropped) contribute zeros — the fixed-budget semantics.
        import torch.distributed as dist
        for p in trainable_params:
            if p.grad is None:
                p.grad = torch.zeros_like(p)
            dist.all_reduce(p.grad, op=dist.ReduceOp.AVG)
    grad_norm = torch.nn.utils.clip_grad_norm_(trainable_params, max_grad_norm)
    gn = grad_norm.item() if hasattr(grad_norm, "item") else float(grad_norm)
    # Guard BEFORE stepping: clip_grad_norm_ does not sanitize nan/inf.
    # Under DDP the grads (hence gn) are identical on every rank, so the
    # skip decision is globally consistent by construction.
    if math.isfinite(gn):
        optim.step()
    else:
        optim.zero_grad(set_to_none=True)
        print(f"[grpo] non-finite grad norm ({gn}) — skipping optimizer step", flush=True)
    metrics = {
        "kl_mean": float(np.mean(sample_kls_log)) if sample_kls_log else 0.0,
        "entropy": float(np.mean(sample_entropy_log)) if sample_entropy_log else 0.0,
    }
    mean_loss = float(np.mean(sample_losses_log)) if sample_losses_log else 0.0
    return mean_loss, gn, metrics


# --------------------------------------------------------------------------------------
# reward


def build_reward_fn(args, tokenizer):
    """Returns (score_fn, floor). score_fn(texts, golds, layers) -> RewardResult-like.

    Default: rl_reward.score_texts against the frozen LC-AR on --reward-device.
    --toy-reward {token_x,constant} replaces it with a deterministic toy (no AR
    loaded, 1 GPU) for the direction/no-op gates.
    """
    from oracle_lens.pipeline import rl_reward

    floor = rl_reward.FAILED_EXTRACTION_REWARD  # -4.0, the cos=-1 floor

    if args.toy_reward != "none":
        target_id = args.toy_token_id

        def score_toy(texts, golds, layers, resp_ids_list):
            n = len(texts)
            if args.toy_reward == "constant":
                r = torch.ones(n)
            else:  # token_x: fraction of response tokens equal to the target id
                r = torch.tensor([
                    sum(1 for t in ids if int(t) == target_id) / max(1, len(ids))
                    for ids in resp_ids_list
                ], dtype=torch.float32)
            return type("R", (), {
                "reward": r, "fve": torch.full((n,), float("nan")),
                "valid": torch.ones(n, dtype=torch.bool),
            })()

        print(f"[reward] TOY mode {args.toy_reward} (token_id={target_id}) — no AR loaded")
        return score_toy, floor

    from oracle_lens.pipeline.ar_loader import load_ladder_whiteners, load_lc_reconstructor

    t0 = time.time()
    # eager=True is LOAD-BEARING: compiled blocks recompile per (batch, width)
    # and on torch 2.9.1 + Triton 3.4 + Hopper a mid-service recompile produced
    # cos=NaN then a CUDA illegal memory access (ar_loader docstring).
    recon = load_lc_reconstructor(
        Path(args.ar_ckpt), device=args.reward_device, eager=True,
    )
    whiteners = load_ladder_whiteners(
        Path(args.whitener_dir), prefix=args.whitener_prefix, ridge_c=args.ridge,
        layers=tuple(recon.layers),
    )
    print(f"[reward] frozen LC-AR on {args.reward_device} "
          f"(layers={list(recon.layers)}) + {len(whiteners)} whiteners "
          f"in {time.time() - t0:.0f}s", flush=True)

    space = rl_reward.RewardSpace(whiten=args.reward_whiten, unit_norm=args.reward_unit_norm)
    print(f"[reward] agg={args.reward_agg} space(whiten={args.reward_whiten}, "
          f"unit_norm={args.reward_unit_norm})"
          + (f" k_max={args.reward_k_max} loo={args.reward_loo}"
             if args.reward_agg in ("joint", "contrastive") else "")
          + (f" contrastive_beta={args.reward_contrastive_beta} "
             f"n_distractors={args.reward_n_distractors}"
             if args.reward_agg == "contrastive" else ""), flush=True)

    # Contrastive distractors: a fixed per-layer bank of gold activations (the RL batch
    # has too few prompts/layer to be its own distractor pool). Whitened once here in the
    # scored space so score_texts_joint can refit rollout bullets straight onto them.
    distractor_bank = None
    if args.reward_agg == "contrastive":
        import pyarrow.parquet as pq
        _tbl = pq.ParquetFile(args.rl_parquet).read(columns=["layer", "gold_vector"])
        _blayers = _tbl.column("layer").to_pylist()
        _bgolds = np.asarray(
            _tbl.column("gold_vector").combine_chunks().flatten(), dtype=np.float32
        ).reshape(len(_blayers), -1)
        _by: dict[int, list[int]] = {}
        for _i, _lay in enumerate(_blayers):
            _by.setdefault(int(_lay), []).append(_i)
        _gen = torch.Generator().manual_seed(0)
        distractor_bank = {}
        for _lay, _idxs in _by.items():
            _gv = torch.from_numpy(_bgolds[_idxs]).float()
            if len(_gv) > args.reward_bank_cap:
                _gv = _gv[torch.randperm(len(_gv), generator=_gen)[: args.reward_bank_cap]]
            _w = whiteners[int(_lay)]
            distractor_bank[int(_lay)] = _w.whiten(_gv) if args.reward_whiten else (_gv - _w.mu)
        print(f"[reward] contrastive distractor bank (cap {args.reward_bank_cap}): "
              f"{ {L: len(v) for L, v in sorted(distractor_bank.items())} }", flush=True)

    def score_ar(texts, golds, layers, resp_ids_list):
        if args.reward_agg in ("joint", "contrastive"):
            return rl_reward.score_texts_joint(
                recon, tokenizer, texts,
                golds=golds, layers=layers, whiteners=whiteners, space=space,
                k_max=args.reward_k_max, loo=args.reward_loo,
                loo_lambda=args.reward_loo_lambda,
                contrastive=(args.reward_contrastive_beta
                             if args.reward_agg == "contrastive" else 0.0),
                n_distractors=args.reward_n_distractors,
                distractor_bank=distractor_bank,
                diversity_lambda=args.diversity_lambda,
                micro_batch=args.rm_micro_batch, device=args.reward_device,
            )
        return rl_reward.score_texts(
            recon, tokenizer, texts,
            golds=golds, layers=layers, whiteners=whiteners, space=space,
            micro_batch=args.rm_micro_batch, device=args.reward_device,
        )

    return score_ar, floor


def ddp_score_on_rank0(score_fn, texts, golds_np, layers, world, rank):
    """Score all ranks' rollouts on the single AR host (rank 0), return this
    rank's slice as a RewardResult-like (reward/fve/valid tensors).

    Object collectives are fine at this scale: texts are strings, golds ~10 MB
    per rank per step.
    """
    import torch.distributed as dist

    is_main = rank == 0
    gathered = [None] * world if is_main else None
    dist.gather_object((texts, golds_np, list(layers)), gathered, dst=0)
    slices_box = [None]
    if is_main:
        texts_all = [t for g in gathered for t in g[0]]
        golds_all = np.concatenate([g[1] for g in gathered])
        layers_all = [ly for g in gathered for ly in g[2]]
        res_all = score_fn(
            texts_all, torch.from_numpy(golds_all).float(),
            torch.tensor(layers_all, dtype=torch.long), [],
        )
        # carry all 5 fields (cos=bullet_diversity, mse=LOO reward) so DDP logs the
        # same metrics as the single-GPU path.
        def _sl(x, a, b):
            return (x[a:b].cpu() if x is not None else None)
        slices, off = [], 0
        for g in gathered:
            n_g = len(g[0])
            slices.append((res_all.reward[off:off + n_g].cpu(),
                           res_all.fve[off:off + n_g].cpu(),
                           _sl(getattr(res_all, "cos", None), off, off + n_g),
                           _sl(getattr(res_all, "mse", None), off, off + n_g),
                           res_all.valid[off:off + n_g].cpu()))
            off += n_g
        slices_box = [slices]
    dist.broadcast_object_list(slices_box, src=0)
    r, f, cs, ms, v = slices_box[0][rank]
    nan = torch.full_like(r, float("nan"))
    return type("R", (), {"reward": r, "fve": f,
                          "cos": cs if cs is not None else nan,
                          "mse": ms if ms is not None else nan, "valid": v})()


# --------------------------------------------------------------------------------------


def main():
    p = argparse.ArgumentParser()
    add_config_arg(p)
    p.add_argument("--ao-lora", required=True,
                   help="SFT AO LoRA adapter dir (the lora_hf/ stripped copy). Loaded as "
                        "the trainable 'default' adapter AND the frozen 'reference' KL anchor.")
    p.add_argument("--base-ckpt", default="Qwen/Qwen3.6-27B")
    p.add_argument("--ar-ckpt", required=True,
                   help="Frozen LC-AR checkpoint dir (lora/ + heads.pt) — the reward model.")
    p.add_argument("--whitener-dir", required=True)
    p.add_argument("--whitener-prefix", default="whitening_iolens_chat")
    p.add_argument("--ridge", type=float, default=0.1)
    p.add_argument("--reward-agg", choices=["single", "joint", "contrastive"], default="single",
                   help="single = whole-text one readout; joint = per-bullet AR + NNLS "
                        "refit of gold onto the bullet images (distill bullets reward); "
                        "contrastive = joint FVE minus beta*max same-layer distractor FVE "
                        "(margin reward -> distinctive, non-generic readouts).")
    p.add_argument("--reward-contrastive-beta", type=float, default=1.0,
                   help="Weight on the distractor term for --reward-agg contrastive.")
    p.add_argument("--reward-n-distractors", type=int, default=8,
                   help="Same-layer distractor activations per rollout (contrastive reward).")
    p.add_argument("--reward-bank-cap", type=int, default=128,
                   help="Max gold activations per layer kept in the contrastive distractor bank.")
    p.add_argument("--reward-whiten", action=argparse.BooleanOptionalAction, default=True,
                   help="score in the per-layer whitened basis (the canonical yardstick).")
    p.add_argument("--reward-unit-norm", action=argparse.BooleanOptionalAction, default=True,
                   help="single: cosine -2(1-cos) vs -MSE; joint: reconstruct from unit "
                        "concept directions vs raw-magnitude bullet images.")
    p.add_argument("--reward-k-max", type=int, default=4,
                   help="joint: cap bullets per rollout (matches the 4-pick distill target; "
                        "prevents gaming the NNLS with many repetitive bullets).")
    p.add_argument("--reward-loo", action=argparse.BooleanOptionalAction, default=False,
                   help="joint: leave-one-out marginal reward (sum_j FVE(all)-FVE(all\\j)) "
                        "-> rewards DISTINCT contributing bullets, kills duplicate-bullet "
                        "degeneration. Off = plain joint FVE. NOTE: PURE LOO mode-collapses "
                        "to one bullet (bullet 1 carries ~all FVE) — use --reward-loo-lambda.")
    p.add_argument("--reward-loo-lambda", type=float, default=0.0,
                   help="joint + lambda*LOO combined reward. The joint term keeps the "
                        "4-bullet format & FVE (blocks the 1-bullet collapse of pure LOO); "
                        "lambda*LOO adds gentle anti-filler pressure. Try 0.1-0.3.")
    p.add_argument("--rl-parquet", required=True)
    p.add_argument("--eval-parquet", default=None,
                   help="Gate parquet for the held-out eval (disjoint by construction at prep).")
    p.add_argument("--diag-parquet", default=None,
                   help="Diagnostic probe bank (build_diag_bank.py: acetaminophen/GFP/"
                        "suppression items). Greedy AO readouts logged as the wandb "
                        "diag/samples table at eval cadence; qualitative only.")
    p.add_argument("--sidecar", required=True, help="nla_meta.yaml (schema v2).")
    p.add_argument("--save-dir", required=True)
    p.add_argument("--actor-device", default="cuda:0")
    p.add_argument("--reward-device", default="cuda:1")
    p.add_argument("--num-steps", type=int, default=200)
    p.add_argument("--batch-prompts", type=int, default=8, help="prompts per step")
    p.add_argument("--group-size", type=int, default=64, help="samples per prompt")
    p.add_argument("--max-new-tokens", type=int, default=96,
                   help="AR reads <=64 bare tokens + <explanation> scaffold + headroom; "
                        "gate-10's known-good numbers were measured at 96.")
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--eval-temperature", type=float, default=None,
                   help="Eval-only sampling temperature (default --temperature; 0 = greedy).")
    p.add_argument("--lr", type=float, default=1.41e-5)
    p.add_argument("--max-grad-norm", type=float, default=1.0)
    p.add_argument("--kl-beta", type=float, default=0.05,
                   help="Coefficient on the EXACT analytic KL(policy||ref). 0.05 = the "
                        "signed-off run100 value for the same KL(policy||ref) quantity.")
    p.add_argument("--kl-beta-final", type=float, default=None,
                   help="lever-2: if set, linearly anneal kl_beta from --kl-beta to this over "
                        "training (loosen the anchor late so it can explore past the SFT format).")
    p.add_argument("--entropy-coef", type=float, default=0.0,
                   help="lever-2: entropy bonus coefficient (adds +coef*H(policy) to the "
                        "objective -> keeps exploration alive against the entropy decay we "
                        "saw). Try 0.001-0.01.")
    p.add_argument("--diversity-lambda", type=float, default=0.0,
                   help="DPP diversity bonus: reward += lambda*det(Gram of bullet AR-image "
                        "directions). Pays for mutually-orthogonal (distinct) bullets on top "
                        "of joint FVE; un-gameable by rephrasing (twins -> det 0). Try 0.05-0.2.")
    p.add_argument("--rloo", action="store_true",
                   help="RLOO advantage: leave-one-out group baseline r_i - mean(r_{-i}) "
                        "(unbiased, lower variance than GRPO group-mean). Takes precedence "
                        "over dr-grpo advantage.")
    p.add_argument("--adv-std-floor", type=float, default=0.0,
                   help="Dr.GRPO clipped std-norm: advantage = (group_r-mu)/max(std, floor). "
                        "Keeps the per-prompt adaptivity that drove the janky FVE, caps the "
                        "std->0 noise blowup. Try 0.05.")
    p.add_argument("--adv-fixed-scale", type=float, default=0.0,
                   help="Dr.GRPO: divide (group_r - mu) by this FIXED constant instead of "
                        "per-group std (recovers gradient magnitude without std->0 blowup). "
                        "0 = raw. Try ~0.1.")
    p.add_argument("--dr-grpo", action="store_true",
                   help="Dr.GRPO: (1) advantage = group-mean baseline only, no /std; (2) loss = "
                        "token-SUM / fixed token budget (n_total*max_new_tokens), not "
                        "per-sample token-mean. Removes the GRPO length bias + the std->0 "
                        "noise amplification.")
    p.add_argument("--length-penalty", type=float, default=0.0,
                   help="HINGED length penalty on the GRPO signal. Default 0 = design of "
                        "record (no shaping).")
    p.add_argument("--length-threshold", type=int, default=0, help="0 => max_new_tokens - 32.")
    p.add_argument("--gradient-checkpointing", action="store_true", default=False)
    p.add_argument("--gen-prompts-per-call", type=int, default=2,
                   help="Same-length prompts batched per generate call (k*G rows, "
                        "zero padding). 2 => 128-row generation batches at G=64.")
    p.add_argument("--logp-micro-batch", type=int, default=8)
    p.add_argument("--rm-micro-batch", type=int, default=32)
    p.add_argument("--save-every", type=int, default=25)
    p.add_argument("--resume-from-lora", type=str, default=None)
    p.add_argument("--start-step", type=int, default=0)
    p.add_argument("--eval-every", type=int, default=5)
    p.add_argument("--eval-n-prompts", type=int, default=128)
    p.add_argument("--eval-gen-batch", type=int, default=32)
    p.add_argument("--max-rows", type=int, default=0, help="cap training rows (0 = all).")
    p.add_argument("--toy-reward", choices=["none", "token_x", "constant"], default="none",
                   help="Deterministic toy reward for the direction/no-op gates.")
    p.add_argument("--toy-token-id", type=int, default=0)
    p.add_argument("--wandb-project", default="ola")
    p.add_argument("--wandb-name", default=None)
    p.add_argument("--wandb-group", default="rl-sc")
    p.add_argument("--wandb-tags", default=None)
    p.add_argument("--no-wandb", action="store_true")
    p.add_argument("--seed", type=int, default=0)
    apply_config_defaults(p)
    args = p.parse_args()

    # ---- DDP (torchrun): N actor ranks, one shared reward AR on rank 0 ----
    # Design: identical data shuffle everywhere; each rank rolls out its
    # batch_prompts/W prompt shard on its own GPU; texts/golds gather to rank 0
    # (the only rank holding the AR), rewards broadcast back; grads all-reduce
    # (AVG) before an identical clip+step on every rank. No DDP wrapper — it
    # fights PEFT adapter switching and the injection hook; the LoRA grads are
    # ~234 MB bf16, one cheap collective. Launch:
    #   torchrun --nproc-per-node=7 train_rl_ao.py ... --reward-device cuda:7
    ddp_world = int(os.environ.get("WORLD_SIZE", "1"))
    ddp_rank = int(os.environ.get("RANK", "0"))
    is_main = ddp_rank == 0
    if ddp_world > 1:
        import torch.distributed as dist
        dist.init_process_group("nccl")
        local_rank = int(os.environ["LOCAL_RANK"])
        args.actor_device = f"cuda:{local_rank}"
        torch.cuda.set_device(local_rank)
        assert args.batch_prompts % ddp_world == 0, (
            f"--batch-prompts {args.batch_prompts} must divide by WORLD_SIZE {ddp_world}"
        )
        if is_main:
            print(f"[ddp] {ddp_world} actor ranks; reward AR on {args.reward_device} "
                  f"(rank 0); {args.batch_prompts // ddp_world} prompts/rank", flush=True)

    # ---- fail-fast checks (BEFORE any model loading) ----
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    _existing_iters = sorted(save_dir.glob("iter_*"))
    if _existing_iters and args.resume_from_lora is None:
        raise SystemExit(
            f"[save] {save_dir} already contains {len(_existing_iters)} iter_* "
            f"checkpoints (latest: {_existing_iters[-1].name}) — refusing to "
            f"overwrite. Resume with --resume-from-lora {_existing_iters[-1]} "
            f"or use a fresh --save-dir."
        )
    # RL checkpoints are self-describing: snapshot the sidecar next to them; on
    # resume ASSERT the tokens/extraction contract still agrees.
    import shutil

    import yaml as _yaml
    _side_src = Path(args.sidecar)
    _side_dst = save_dir / "nla_meta.yaml"
    if _side_dst.exists():
        _prev = _yaml.safe_load(_side_dst.read_text())
        _cur = _yaml.safe_load(_side_src.read_text())
        for _k in ("tokens", "extraction"):
            assert _prev.get(_k) == _cur.get(_k), (
                f"save-dir sidecar snapshot disagrees with --sidecar on {_k!r} — this run "
                f"would score/inject differently than the checkpoints it resumes."
            )
    elif is_main:  # single writer under DDP; non-main never reads it
        shutil.copy2(_side_src, _side_dst)

    # On-policy consistency (source rationale, kept verbatim): at T=1 sampling
    # and scoring distributions match; any other T silently biases the gradient.
    assert args.temperature == 1.0, (
        f"--temperature {args.temperature} != 1.0: rollouts would be sampled from a "
        f"distribution the on-policy update does not score under."
    )
    if args.length_threshold <= 0:
        args.length_threshold = max(1, args.max_new_tokens - 32)
    if args.length_penalty > 0:
        print(f"[len] hinged penalty {args.length_penalty}/token past "
              f"{args.length_threshold} tokens (cap {args.max_new_tokens})", flush=True)

    # Rollout sampling must DIFFER per rank (independent explorations of each
    # rank's prompts); the data-order rng below stays seed-identical everywhere.
    torch.manual_seed(args.seed * 1000 + ddp_rank)
    np.random.seed(args.seed)
    device = args.actor_device

    # ---- tokenizer + sidecar (tokenizer-drift tripwires) ----
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.base_ckpt)
    cfg = load_sidecar(args.sidecar, tokenizer)
    inj_id = cfg.injection_token_id
    left_id = cfg.left_neighbor_id
    right_id = cfg.right_neighbor_id
    print(f"[cfg] inj={cfg.injection_char!r} id={inj_id} neighbors=({left_id},{right_id}) "
          f"d_model={cfg.d_model}", flush=True)

    # ---- actor: bf16 base + AO-SFT LoRA ("default", trainable) + frozen
    #      "reference" adapter (= AO-SFT init) as the KL anchor. ----
    from peft import PeftModel

    t0 = time.time()
    print(f"[actor] base={args.base_ckpt} + AO-LoRA={args.ao_lora} (bf16, {device})",
          flush=True)
    base = AutoModelForCausalLM.from_pretrained(
        args.base_ckpt, dtype=torch.bfloat16, attn_implementation="sdpa",
    ).to(device)
    policy_ckpt = args.resume_from_lora or args.ao_lora
    if args.resume_from_lora:
        print(f"[actor] RESUMING policy LoRA from {args.resume_from_lora} "
              f"(KL reference stays {args.ao_lora})", flush=True)
    actor = PeftModel.from_pretrained(
        base, policy_ckpt, adapter_name="default", is_trainable=True,
    )
    actor.load_adapter(args.ao_lora, adapter_name="reference")  # frozen KL anchor
    actor.set_adapter("default")

    # Liveness assert (inert-LoRA guard): every saved adapter tensor must have
    # found a module. PEFT's load_adapter never raises on missing keys, so a
    # key-schema drift (e.g. compiled `_orig_mod.` names) silently loads NOTHING
    # and the "policy" would be the bare SFT-less base.
    from safetensors.torch import load_file as _load_file
    _saved = _load_file(str(Path(args.ao_lora) / "adapter_model.safetensors"))
    _model_keys = {n for n, _ in actor.named_parameters() if ".default." in n}
    _mapped = 0
    for k in _saved:
        # saved: base_model.model...lora_A.weight -> live: ...lora_A.default.weight
        want = k.replace(".lora_A.", ".lora_A.default.").replace(".lora_B.", ".lora_B.default.")
        if not want.startswith("base_model."):
            want = "base_model.model." + want
        if want in _model_keys:
            _mapped += 1
    _live_frac = _mapped / max(1, len(_saved))
    assert _live_frac >= 0.9, (
        f"inert LoRA: only {_mapped}/{len(_saved)} saved adapter tensors map onto the "
        f"live model ({_live_frac:.0%}) — key-schema drift (compiled ckpt? PEFT version?)."
    )
    print(f"[actor] adapter liveness: {_mapped}/{len(_saved)} tensors mapped", flush=True)

    # Init sanity: fresh start loads the SAME ckpt twice -> diff must be exactly 0.
    _diff = 0.0
    _sd = actor.state_dict()
    for n in list(_sd):
        if ".default." in n and "lora_" in n:
            _ref = _sd.get(n.replace(".default.", ".reference."))
            if _ref is not None:
                _diff += (_sd[n].float() - _ref.float()).pow(2).sum().item()
    del _sd
    print(f"[actor] sum((lora_default - lora_reference)^2) = {_diff:.3e}", flush=True)
    if args.resume_from_lora is None:
        assert _diff == 0.0, "fresh start: default and reference must be identical"
    elif _diff == 0.0:
        print("[actor] WARNING: resumed adapter IDENTICAL to reference — untrained "
              "resume dir or silent load failure.", flush=True)

    # Pin the reference adapter frozen (PEFT's set_adapter toggles trainability).
    for _n, _p in actor.named_parameters():
        if ".reference." in _n:
            _p.requires_grad_(False)
    actor.print_trainable_parameters()
    actor.train()
    if args.gradient_checkpointing:
        actor.gradient_checkpointing_enable()
        actor.enable_input_require_grads()
        print("[actor] gradient_checkpointing ENABLED", flush=True)
    print(f"[actor] loaded in {time.time() - t0:.0f}s", flush=True)

    # ---- injection hook: embedding REPLACEMENT at the marker ----
    vectors_ref = [None]
    register_embed_injection_hook(actor, vectors_ref, inj_id, left_id, right_id)

    # ---- stop ids: tokenizer eos u generation_config eos u THE MARKER (bug 2) ----
    eos_ids = {tokenizer.eos_token_id}
    _gc_eos = getattr(getattr(actor, "generation_config", None), "eos_token_id", None)
    if _gc_eos is not None:
        eos_ids.update(_gc_eos if isinstance(_gc_eos, (list, tuple)) else [_gc_eos])
    eos_ids.discard(None)
    eos_ids.add(inj_id)
    pad_id = tokenizer.eos_token_id
    print(f"[rollout] stop ids: {sorted(eos_ids)} (marker {inj_id} is a stop token)",
          flush=True)

    # ---- reward (frozen LC-AR on --reward-device, or toy) ----
    # Under DDP only rank 0 hosts the AR (toy rewards stay rank-local).
    if ddp_world > 1 and not is_main and args.toy_reward == "none":
        from oracle_lens.pipeline import rl_reward as _rlr
        score_fn, floor = None, _rlr.FAILED_EXTRACTION_REWARD
    else:
        score_fn, floor = build_reward_fn(args, tokenizer)

    # ---- data ----
    rows = load_ao_rl_dataset(
        args.rl_parquet, inj_id=inj_id, n_max=(args.max_rows or None),
    )
    print(f"[data] {len(rows)} train rows from {args.rl_parquet}", flush=True)
    eval_rows = []
    if args.eval_every > 0 and args.eval_n_prompts > 0:
        assert args.eval_parquet, "--eval-parquet required when evals are enabled"
        eval_rows = load_ao_rl_dataset(
            args.eval_parquet, inj_id=inj_id, n_max=args.eval_n_prompts,
        )
        print(f"[data] {len(eval_rows)} eval rows from {args.eval_parquet} "
              f"(disjoint at prep)", flush=True)
    diag_rows = []
    if args.diag_parquet and is_main:
        diag_rows = load_ao_rl_dataset(args.diag_parquet, inj_id=inj_id)
        print(f"[data] {len(diag_rows)} diagnostic probe rows from {args.diag_parquet}",
              flush=True)
    if not is_main:
        # non-main ranks: no wandb, no eval/diag generation, no checkpoint writes
        args.no_wandb = True
        eval_rows = []

    # ---- optimizer (AdamW; bnb 8-bit if present, else torch — LoRA-scale states) ----
    try:
        import bitsandbytes as bnb
        _adam_cls = bnb.optim.AdamW8bit
        print(f"[optim] bitsandbytes AdamW8bit (bnb {bnb.__version__})")
    except ImportError:
        _adam_cls = torch.optim.AdamW
        print("[optim] torch AdamW (fp32 moments — fine at LoRA scale)")
    trainable = [p for p in actor.parameters() if p.requires_grad]
    optim = _adam_cls(trainable, lr=args.lr, betas=(0.9, 0.95), weight_decay=0.0)

    if args.resume_from_lora is not None:
        _opt_ckpt = find_optim_ckpt(args.save_dir, args.resume_from_lora)
        if _opt_ckpt is not None:
            print(f"[resume] optimizer state: {_opt_ckpt}", flush=True)
            _opt_st = torch.load(str(_opt_ckpt), map_location="cpu", weights_only=True)
            _saved_step = int(_opt_st.get("step", 0))
            if args.start_step == 0 and _saved_step > 0:
                args.start_step = _saved_step
                print(f"[resume] --start-step defaulted to saved step {_saved_step}",
                      flush=True)
            elif args.start_step != _saved_step:
                print(f"[resume] WARN: --start-step {args.start_step} != saved "
                      f"{_saved_step} — trusting the flag.", flush=True)
            try:
                optim.load_state_dict(_opt_st["actor_optim"])
                print("[resume] optimizer state restored", flush=True)
            except (ValueError, KeyError, RuntimeError) as _e:
                print(f"[resume] WARN: optimizer state incompatible ({_e}) — "
                      f"Adam moments restart.", flush=True)
        else:
            warn_cold_adam(args.start_step)

    if is_main:
        save_resolved_config(args, args.save_dir)

    # ---- wandb ----
    import wandb
    if not args.no_wandb:
        tags = (args.wandb_tags.split(",") if args.wandb_tags else []) + ["self-contained"]
        wandb.init(project=args.wandb_project, name=args.wandb_name,
                   group=args.wandb_group, tags=tags, config=vars(args))

    from oracle_lens.pipeline.rl_reward import extract_explanation, split_bullets

    rng = np.random.default_rng(args.seed)
    pending_idxs = list(range(len(rows)))
    rng.shuffle(pending_idxs)
    cursor = 0
    if args.start_step > 0:
        for _ in range(args.start_step):
            if cursor + args.batch_prompts > len(pending_idxs):
                rng.shuffle(pending_idxs)
                cursor = 0
            cursor += args.batch_prompts
        print(f"[data] fast-forwarded cursor through {args.start_step} steps", flush=True)

    eval_table_data = []

    for step in range(args.start_step, args.num_steps):
        t0 = time.time()
        if cursor + args.batch_prompts > len(pending_idxs):
            rng.shuffle(pending_idxs)
            cursor = 0
        batch_idxs = pending_idxs[cursor: cursor + args.batch_prompts]
        cursor += args.batch_prompts
        if ddp_world > 1:
            # identical shuffle everywhere; each rank rolls out its own shard
            _bp_local = args.batch_prompts // ddp_world
            batch_idxs = batch_idxs[ddp_rank * _bp_local: (ddp_rank + 1) * _bp_local]

        # ---- rollouts ----
        actor.eval()
        all_full_ids = []
        all_resp_ids = []
        all_prompt_lens = []
        all_golds = []
        all_injects = []
        all_layers = []
        all_texts = []
        all_prompt_group = []
        all_resp_lens = []
        all_row_ids = []
        all_sources = []
        # Bucket by exact prompt length, then chunk k prompts per generate call
        # (all rows in a call share one length -> zero padding, GDN-safe; the
        # hook injects each prompt's vector at its own row's marker).
        by_len: dict[int, list[int]] = {}
        for gi, row_idx in enumerate(batch_idxs):
            by_len.setdefault(len(rows[row_idx]["prompt_ids"]), []).append(gi)
        for _plen, gis in sorted(by_len.items()):
            for c0 in range(0, len(gis), args.gen_prompts_per_call):
                chunk_gis = gis[c0: c0 + args.gen_prompts_per_call]
                chunk_rows = [rows[batch_idxs[gi]] for gi in chunk_gis]
                per_prompt = rollout_prompts(
                    actor,
                    [r["prompt_ids"] for r in chunk_rows],
                    [torch.from_numpy(r["inject"]) for r in chunk_rows],
                    vectors_ref, args.group_size, args.max_new_tokens,
                    args.temperature, device, eos_ids, pad_id,
                )
                for gi, row, responses in zip(chunk_gis, chunk_rows, per_prompt, strict=True):
                    inject_vec = torch.from_numpy(row["inject"])
                    for r in responses:
                        all_full_ids.append(r["full_ids"])
                        all_resp_ids.append(r["resp_ids"])
                        all_prompt_lens.append(r["prompt_len"])
                        all_golds.append(row["gold"])
                        all_injects.append(inject_vec)
                        all_layers.append(row["layer"])
                        all_texts.append(
                            tokenizer.decode(r["resp_ids"], skip_special_tokens=True))
                        all_prompt_group.append(gi)
                        all_resp_lens.append(int(r["n_resp"]))
                        all_row_ids.append(row["row_id"])
                        all_sources.append(row.get("source_text", ""))
        t_gen_end = time.time()
        n_rollouts = len(all_full_ids)

        # ---- injection-success mask: FULL-SEQUENCE neighbor-valid site count == 1
        # (bug-2 hardening; the hook's own criterion, so pre-check and count assert
        # agree by construction — a marker-STOPPED rollout keeps count 1 and stays
        # in training, like an eos). No CJK proxy (mechanism check supersedes the
        # erodible output-symptom check). ----
        marker_ok = [
            count_valid_sites(all_full_ids[i].tolist(), inj_id, left_id, right_id) == 1
            for i in range(n_rollouts)
        ]
        inject_ok = marker_ok
        n_marker_bad = int(sum(1 for m in marker_ok if not m))
        inject_ok_t = torch.tensor(inject_ok, dtype=torch.bool, device=device)
        # Truncated = hit the cap AND never emitted a stop id. A marker-stopped
        # rollout ends in inj_id ∈ eos_ids -> classified STOPPED, scored normally.
        truncated = [
            (all_full_ids[i].numel() - all_prompt_lens[i] >= args.max_new_tokens)
            and (int(all_full_ids[i][-1]) not in eos_ids)
            for i in range(n_rollouts)
        ]
        n_truncated = int(sum(truncated))

        # ---- scoring: frozen LC-AR (whitened cosine, r ∈ [-4, 0]) ----
        expl_texts = [extract_explanation(t) for t in all_texts]  # "" on miss
        if ddp_world > 1 and args.toy_reward == "none":
            res = ddp_score_on_rank0(
                score_fn, expl_texts, np.stack(all_golds), all_layers,
                ddp_world, ddp_rank,
            )
        else:
            res = score_fn(
                expl_texts,
                torch.from_numpy(np.stack(all_golds)).float(),
                torch.tensor(all_layers, dtype=torch.long),
                all_resp_ids,
            )
        rewards_raw = res.reward.clone()  # pure reconstruction signal, feeds FVE logging
        valid_mask = res.valid.clone().bool()
        # Truncated -> floor, still trained on (the anti-runaway gradient) and
        # still in the group baseline. Floor = -4.0 (cos=-1), OUR scale — the
        # source's -2.0 fill would rank failures above bad-but-valid text.
        # TOY mode keeps rewards pure: the direction/no-op gates probe the
        # optimizer mechanism, and the floor would contaminate them (constant
        # mode stops being constant the moment one rollout truncates).
        apply_trunc_floor = args.toy_reward == "none"
        rewards_filled = [
            floor if (apply_trunc_floor and truncated[i]) else float(rewards_raw[i])
            for i in range(n_rollouts)
        ]
        rewards_t = torch.tensor(rewards_filled, dtype=torch.float32, device=device)
        t_score_end = time.time()

        # ---- reward shaping (length penalty; default 0 = no-op) ----
        shape_terms = {}
        if args.length_penalty > 0:
            n_tok = torch.tensor(all_resp_lens, dtype=torch.float32, device=device)
            overage = (n_tok - float(args.length_threshold)).clamp_min(0.0)
            rewards_t = rewards_t - args.length_penalty * overage
            shape_terms["av/len_pen_mean"] = (args.length_penalty * overage).mean().item()
            shape_terms["av/len_overage_frac"] = float((overage > 0).float().mean())
        shape_terms["av/truncated_count"] = n_truncated

        # ---- GRPO group-relative advantage (per-prompt mean & std) ----
        group_t = torch.tensor(all_prompt_group, dtype=torch.long, device=device)
        adv = torch.zeros_like(rewards_t)
        group_stds = []  # within-prompt reward spread: the actual GRPO learning signal
        for gi in range(args.batch_prompts):
            mask = (group_t == gi) & inject_ok_t
            if mask.sum() == 0:
                continue
            group_r = rewards_t[mask]
            mu = group_r.mean()
            sd = group_r.std() if group_r.numel() > 1 else torch.tensor(1.0, device=device)
            # Dr.GRPO advantage: group-mean baseline, normalized by a FIXED constant
            # (--adv-fixed-scale) rather than the per-group std — recovers gradient
            # magnitude (the std used to amplify ~10-30x) WITHOUT the std->0 noise
            # blowup. --adv-fixed-scale 0 = raw (group_r - mu). Legacy = /std.
            # RLOO: leave-one-out baseline (unbiased, low-var)
            if args.rloo and group_r.numel() > 1:
                n_grp = group_r.numel()
                adv[mask] = group_r - (group_r.sum() - group_r) / (n_grp - 1)
            elif args.dr_grpo:
                if args.adv_std_floor > 0:      # clipped std-norm: keep per-prompt adaptivity,
                    adv[mask] = (group_r - mu) / torch.clamp(  # cap std->0 blowup
                        sd, min=args.adv_std_floor)
                elif args.adv_fixed_scale > 0:
                    adv[mask] = (group_r - mu) / args.adv_fixed_scale
                else:
                    adv[mask] = (group_r - mu)
            else:
                adv[mask] = (group_r - mu) / (sd + 1e-6)
            if group_r.numel() > 1:
                group_stds.append(float(sd))
        # A within-group std near 0 means the G rollouts of a prompt all reconstruct
        # equally well -> advantages are noise/floor -> no signal (the per-prompt
        # face of the reward ceiling). Watch av/group_reward_std_mean.
        shape_terms["av/group_reward_std_mean"] = (
            float(np.mean(group_stds)) if group_stds else 0.0)
        shape_terms["av/group_reward_std_min"] = (
            float(np.min(group_stds)) if group_stds else 0.0)
        shape_terms["av/group_frac_degenerate"] = (
            float(np.mean([s < 0.05 for s in group_stds])) if group_stds else 0.0)

        # ---- GRPO update ----
        keep = [i for i, ok in enumerate(inject_ok) if ok]
        if not keep:
            print(f"step {step}: all {n_rollouts} rollouts failed the marker check — "
                  f"skipping update.", flush=True)
            continue
        upd_full_ids = [all_full_ids[i] for i in keep]
        upd_prompt_lens = [all_prompt_lens[i] for i in keep]
        upd_injects = [all_injects[i] for i in keep]
        upd_adv = adv.index_select(0, torch.tensor(keep, device=device))
        actor.train()
        # lever-2 KL anneal: linearly interpolate kl_beta -> kl_beta_final over training
        kl_beta_step = args.kl_beta
        if args.kl_beta_final is not None and args.num_steps > 1:
            frac = step / (args.num_steps - 1)
            kl_beta_step = args.kl_beta + frac * (args.kl_beta_final - args.kl_beta)
        mean_loss_val, grad_norm_val, grpo_metrics = grpo_update_microbatched(
            actor, optim, upd_full_ids, upd_prompt_lens, upd_injects,
            upd_adv, vectors_ref, device, pad_id,
            micro_batch=args.logp_micro_batch,
            kl_beta=kl_beta_step,
            entropy_coef=args.entropy_coef,
            max_grad_norm=args.max_grad_norm,
            n_total=n_rollouts,  # fixed budget: dropped rollouts act as zeros
            ddp_world=ddp_world,
            dr_grpo=args.dr_grpo, gen_len=args.max_new_tokens,
        )
        t_upd_end = time.time()
        if not math.isfinite(mean_loss_val):
            print(f"step {step}: loss={mean_loss_val} non-finite "
                  f"(kl={grpo_metrics.get('kl_mean')}).", flush=True)
            continue

        # ---- logging ----
        valid_np = valid_mask.numpy()
        n_valid = int(valid_np.sum())
        extraction_rate = n_valid / n_rollouts if n_rollouts else 0.0
        frac_cut_off = float(np.mean(truncated)) if truncated else 0.0
        if frac_cut_off > 0.02:
            print(f"[WARN step {step}] {frac_cut_off:.0%} of rollouts truncated at "
                  f"max_new_tokens={args.max_new_tokens} — raise the cap if persistent.",
                  flush=True)
        _fve_valid = res.fve[valid_mask] if n_valid else torch.tensor([float("nan")])
        _r_valid = rewards_raw[valid_mask] if n_valid else torch.tensor([float("nan")])
        log = {
            # headline: FVE = cos^2_w over valid rollouts (SFT baseline 0.102,
            # frozen-AR ceiling 0.1437)
            "fve_pct": float(torch.nanmean(_fve_valid)) * 100.0,
            "wall_s": time.time() - t0,
            "av/grad_norm": grad_norm_val,
            "av/kl_to_ref": grpo_metrics.get("kl_mean", 0.0),
            "av/entropy": grpo_metrics.get("entropy", 0.0),
            "av/advantage_mean": adv.mean().item(),
            "av/advantage_std": adv.std().item(),
            "av/extraction_rate": extraction_rate,
            "av/marker_bad_count": n_marker_bad,
            "av/resp_len": float(np.mean(all_resp_lens)),
            "av/frac_cut_off": frac_cut_off,
            # mean bullets per readout — the LOO-collapse detector (LOO's optimum can
            # drift to a single dominant bullet since bullet 1 carries ~all the FVE).
            "av/n_bullets": float(np.mean(
                [len(split_bullets(t, args.reward_k_max))
                 for t in all_texts])) if all_texts else 0.0,
            "reward/mean": float(rewards_t.mean()),
            "reward/mean_valid": float(_r_valid.mean()),
            "reward/std": float(rewards_t.std()),
            "reward/min": float(rewards_t.min()),
            "reward/max": float(rewards_t.max()),
            # joint mode stashes per-rollout bullet diversity (mean pairwise 1-cos of
            # bullet images) in res.cos; ~0 = duplicate bullets, ~1 = orthogonal.
            "av/bullet_diversity": (
                float(torch.nanmean(res.cos[valid_mask]))
                if (args.reward_agg in ("joint", "contrastive") and n_valid) else float("nan")),
            # both reward channels, logged in EVERY joint run (fve=plain joint,
            # res.mse=LOO sum) so the two are directly comparable regardless of target.
            "reward/joint_fve": (
                float(torch.nanmean(res.fve[valid_mask]))
                if (args.reward_agg in ("joint", "contrastive") and n_valid) else float("nan")),
            "reward/loo": (
                float(torch.nanmean(res.mse[valid_mask]))
                if (args.reward_agg == "joint" and n_valid) else float("nan")),
            # contrastive: res.mse carries the max same-layer distractor FVE (how generic
            # the bullets are). The margin actually optimized = joint_fve - beta*distractor.
            "reward/distractor_fve": (
                float(torch.nanmean(res.mse[valid_mask]))
                if (args.reward_agg == "contrastive" and n_valid) else float("nan")),
            "time/gen_s": t_gen_end - t0,
            "time/score_s": t_score_end - t_gen_end,
            "time/update_s": t_upd_end - t_score_end,
            "rollout/gen_tok_per_s": sum(all_resp_lens) / max(1e-6, t_gen_end - t0),
            "rollout/n_rollouts": float(n_rollouts),
        }
        log.update(shape_terms)
        if is_main:
            print(
                f"step {step:04d} | r {log['reward/mean']:.3f} "
                f"(valid {log['reward/mean_valid']:.3f}) "
                f"| FVE {log['fve_pct']:.1f}% | kl {log['av/kl_to_ref']:.4f} "
                f"| ent {log['av/entropy']:.3f} | ext {extraction_rate:.0%} "
                f"| nbul {log['av/n_bullets']:.2f} "
                f"| len {log['av/resp_len']:.0f} cut {log.get('av/frac_cut_off', 0):.0%} "
                f"| gnorm {grad_norm_val:.3f} "
                f"| gstd {log.get('av/group_reward_std_mean', 0):.3f} "
                f"| t {log['wall_s']:.0f}s "
                f"(gen {log['time/gen_s']:.0f} score {log['time/score_s']:.0f} "
                f"upd {log['time/update_s']:.0f})",
                flush=True,
            )

        # ---- train rollout samples -> wandb (eval-step cadence; 2 per group) ----
        # source_text = the crop span whose preceding residual is the activation:
        # lets the wandb table show (source, injected-layer, AO explanation, reward).
        if (not args.no_wandb and args.eval_every > 0 and step % args.eval_every == 0):
            _seen_groups: dict[int, int] = {}
            _tbl = []
            for i in range(n_rollouts):
                gi = all_prompt_group[i]
                if _seen_groups.get(gi, 0) >= 2:
                    continue
                _seen_groups[gi] = _seen_groups.get(gi, 0) + 1
                _tbl.append([
                    step, all_row_ids[i], all_layers[i],
                    (all_sources[i] or "")[:300],
                    all_texts[i][:600],
                    len(split_bullets(all_texts[i], args.reward_k_max)),
                    round(float(rewards_t[i]), 4),
                    round(float(res.fve[i]), 4) if bool(valid_mask[i]) else float("nan"),
                    bool(truncated[i]),
                ])
            log["train/samples"] = wandb.Table(
                columns=["step", "row_id", "layer", "source_text", "response",
                         "n_bullets", "reward", "fve", "truncated"],
                data=_tbl,
            )

        # ---- held-out eval: LENGTH-BUCKETED unpadded generation (bug-13 fix) ----
        if args.eval_every > 0 and eval_rows and step % args.eval_every == 0:
            actor.eval()
            _t_eval = time.time()
            _et = (args.eval_temperature if args.eval_temperature is not None
                   else args.temperature)
            # Bucket rows by exact prompt length -> every generate() batch is
            # unpadded. Our prompts are one template varying only in the layer
            # token, so this is 1-2 buckets in practice.
            _by_len: dict[int, list[int]] = {}
            for ei, r in enumerate(eval_rows):
                _by_len.setdefault(len(r["prompt_ids"]), []).append(ei)
            _resp_text = [None] * len(eval_rows)
            with torch.no_grad():
                for _plen, _eis in sorted(_by_len.items()):
                    for c0 in range(0, len(_eis), args.eval_gen_batch):
                        chunk = _eis[c0: c0 + args.eval_gen_batch]
                        ids = torch.tensor(
                            [eval_rows[ei]["prompt_ids"] for ei in chunk],
                            dtype=torch.long, device=device,
                        )
                        vectors_ref[0] = torch.stack([
                            torch.from_numpy(eval_rows[ei]["inject"]) for ei in chunk
                        ]).to(device).float()
                        try:
                            gen = actor.generate(
                                input_ids=ids,
                                attention_mask=torch.ones_like(ids),
                                max_new_tokens=args.max_new_tokens,
                                do_sample=(_et > 0),
                                **({"temperature": _et} if _et > 0 else {}),
                                top_p=1.0, top_k=0, repetition_penalty=1.0,
                                pad_token_id=pad_id,
                                eos_token_id=sorted(eos_ids),
                                return_dict_in_generate=True,
                            )
                        finally:
                            vectors_ref[0] = None
                        new_toks = gen.sequences[:, ids.shape[1]:]
                        for j, ei in enumerate(chunk):
                            _rids = new_toks[j].tolist()
                            _n_real = next(
                                (k + 1 for k, t in enumerate(_rids) if t in eos_ids),
                                len(_rids),
                            )
                            _resp_text[ei] = tokenizer.decode(
                                _rids[:_n_real], skip_special_tokens=True)
            # Batched scoring through the same reward path as training.
            _e_expl = [extract_explanation(t or "") for t in _resp_text]
            _e_res = score_fn(
                _e_expl,
                torch.from_numpy(np.stack([r["gold"] for r in eval_rows])).float(),
                torch.tensor([r["layer"] for r in eval_rows], dtype=torch.long),
                [[] for _ in eval_rows],
            )
            _e_valid = _e_res.valid.bool()
            _n_ev = int(_e_valid.sum())
            log["eval/reward_mean"] = float(_e_res.reward.mean())
            log["eval/reward_mean_valid"] = (
                float(_e_res.reward[_e_valid].mean()) if _n_ev else float("nan"))
            log["eval/fve"] = (
                float(torch.nanmean(_e_res.fve[_e_valid])) if _n_ev else float("nan"))
            log["eval/extraction_rate"] = _n_ev / max(1, len(eval_rows))
            log["time/eval_s"] = time.time() - _t_eval
            for ei in range(len(eval_rows)):
                eval_table_data.append([
                    step, ei, eval_rows[ei]["layer"],
                    (eval_rows[ei].get("source_text") or "")[:300],
                    len(split_bullets(_resp_text[ei] or "", args.reward_k_max)),
                    float(_e_res.reward[ei]),
                    float(_e_res.fve[ei]), bool(_e_valid[ei]),
                    (_e_expl[ei] or "<extraction failed>")[:600],
                ])
            if not args.no_wandb:
                log["eval/samples"] = wandb.Table(
                    columns=["step", "idx", "layer", "source_text", "n_bullets",
                             "reward", "fve", "extracted", "explanation"],
                    data=list(eval_table_data),
                )
            print(f"  [eval@{step}] reward {log['eval/reward_mean']:.3f} "
                  f"| FVE {log['eval/fve']:.4f} "
                  f"| ext {log['eval/extraction_rate']:.0%} "
                  f"| {log['time/eval_s']:.0f}s", flush=True)
            for _ei in (0, 7, 14):
                if _ei < len(eval_rows) and _resp_text[_ei]:
                    _e = (_e_expl[_ei] or "")[:200].replace("\n", " ")
                    print(f"    [eval@{step} idx={_ei} r={float(_e_res.reward[_ei]):.3f}] {_e}",
                          flush=True)

        # ---- diagnostic probes (acetaminophen/GFP/suppression …): AO readouts on
        # hand-authored situations, logged as a wandb table. Rewards are qualitative
        # context (OOD vs pool crops) — the point is WHAT the AO verbalizes (does the
        # latent danger/GFP/suppressed concept surface?). Sampled at the eval
        # temperature (T=1 by default) to match the training/eval distribution — the
        # on-policy policy is a sampler, so greedy would read off-distribution. ----
        if args.eval_every > 0 and diag_rows and step % args.eval_every == 0:
            actor.eval()
            _t_diag = time.time()
            _dt = (args.eval_temperature if args.eval_temperature is not None
                   else args.temperature)
            _d_text = [None] * len(diag_rows)
            with torch.no_grad():
                for c0 in range(0, len(diag_rows), args.eval_gen_batch):
                    chunk = list(range(c0, min(c0 + args.eval_gen_batch, len(diag_rows))))
                    ids = torch.tensor(
                        [diag_rows[di]["prompt_ids"] for di in chunk],
                        dtype=torch.long, device=device,
                    )
                    vectors_ref[0] = torch.stack([
                        torch.from_numpy(diag_rows[di]["inject"]) for di in chunk
                    ]).to(device).float()
                    try:
                        gen = actor.generate(
                            input_ids=ids, attention_mask=torch.ones_like(ids),
                            max_new_tokens=args.max_new_tokens, do_sample=(_dt > 0),
                            **({"temperature": _dt} if _dt > 0 else {}),
                            top_p=1.0, top_k=0, repetition_penalty=1.0,
                            pad_token_id=pad_id, eos_token_id=sorted(eos_ids),
                            return_dict_in_generate=True,
                        )
                    finally:
                        vectors_ref[0] = None
                    new_toks = gen.sequences[:, ids.shape[1]:]
                    for j, di in enumerate(chunk):
                        _rids = new_toks[j].tolist()
                        _n = next((k + 1 for k, t in enumerate(_rids) if t in eos_ids),
                                  len(_rids))
                        _d_text[di] = tokenizer.decode(_rids[:_n], skip_special_tokens=True)
            _d_expl = [extract_explanation(t or "") for t in _d_text]
            _d_res = score_fn(
                _d_expl,
                torch.from_numpy(np.stack([r["gold"] for r in diag_rows])).float(),
                torch.tensor([r["layer"] for r in diag_rows], dtype=torch.long),
                [[] for _ in diag_rows],
            )
            if not args.no_wandb:
                log["diag/samples"] = wandb.Table(
                    columns=["step", "probe", "layer", "reward", "readout"],
                    data=[[step, diag_rows[di]["source_text"][:200],
                           diag_rows[di]["layer"], round(float(_d_res.reward[di]), 3),
                           (_d_expl[di] or "")[:400]] for di in range(len(diag_rows))],
                )
            log["diag/reward_mean"] = float(_d_res.reward.mean())
            log["time/diag_s"] = time.time() - _t_diag
            # per-probe concept-hit rates (overdose/blackmail/gfp/...): does the readout
            # surface the latent concept? rate = fraction of the probe's layers that hit.
            _hits: dict[str, list[int]] = {}
            for di in range(len(diag_rows)):
                _lab = diag_rows[di]["source_text"].split("]")[0].lstrip("[")
                _kws = _DIAG_HIT_KEYWORDS.get(_lab)
                if not _kws:
                    continue
                _r = (_d_text[di] or "").lower()
                _hits.setdefault(_lab, []).append(int(any(k in _r for k in _kws)))
            for _lab, _hl in _hits.items():
                log[f"diag/hit/{_lab}"] = float(np.mean(_hl))
            if _hits:
                log["diag/hit_rate"] = float(np.mean([np.mean(v) for v in _hits.values()]))
                _hl_str = " ".join(
                    f"{_lab.split('-')[0]} {np.mean(_hv):.2f}"
                    for _lab, _hv in list(_hits.items())[:4])
                print(f"    [diag@{step} hits] {_hl_str} | mean {log['diag/hit_rate']:.2f}",
                      flush=True)
            for di in range(0, len(diag_rows), max(1, len(diag_rows) // 3)):
                _lbl = diag_rows[di]["source_text"].split("]")[0].lstrip("[")
                print(f"    [diag@{step} {_lbl} L{diag_rows[di]['layer']} "
                      f"r={float(_d_res.reward[di]):.3f}] "
                      f"{(_d_expl[di] or '')[:120].replace(chr(10), ' ')}", flush=True)

        if not args.no_wandb:
            wandb.log(log, step=step)

        # ---- save LoRA periodically (+ atomic optimizer state; rank 0 only —
        # weights are identical on every rank by the all-reduce construction) ----
        if is_main and (step + 1) % args.save_every == 0:
            out_dir = save_dir / f"iter_{step + 1:06d}"
            out_dir.mkdir(parents=True, exist_ok=True)
            actor.save_pretrained(str(out_dir))
            _opt_tmp = save_dir / "optim_latest.pt.tmp"
            _opt_dst = save_dir / "optim_latest.pt"
            torch.save({"step": step + 1, "actor_optim": optim.state_dict()}, str(_opt_tmp))
            os.replace(str(_opt_tmp), str(_opt_dst))
            print(f"[save] LoRA -> {out_dir} (+ optim_latest)", flush=True)

    print("done." if is_main else f"done (rank {ddp_rank}).")
    if not args.no_wandb:
        wandb.finish()
    if ddp_world > 1:
        import torch.distributed as dist
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
