"""Per-position AO readout: forward text, read the layer-``L`` residual at every position, and
verbalize each with the Activation Oracle.

This is the engine behind the interactive viewer. For an input text we run ONE base-model forward
(the residual at each position is the "real activation" ``h_p``, captured exactly as the training
dump does — ``ActivationRecorder`` at layer ``L``), then for each position we inject
``inject_rep(h_p)`` into the AO's ``<concept>`` slot and sample ``k`` short descriptions. Those
samples are the "top phrases" the viewer paints — a generative oracle's analogue of a ranked bank.

Capture and generation are split into two functions so the viewer can serve generation on a fast
backend (the verified SGLang input-embeds path) while capture stays on an HF handle — SGLang does
not expose per-layer residuals.
"""

from collections import Counter
from dataclasses import dataclass
from typing import Any

import torch
from jaxtyping import Float
from torch import Tensor

from oracle_lens.pipeline.inject import Rep, inject_rep
from oracle_lens.pipeline.verbalizer import WVPrompt

__all__ = [
    "PositionReadout",
    "aggregate_topk",
    "capture_residuals",
    "generate_phrases",
    "readout_text",
]


@dataclass(frozen=True)
class PositionReadout:
    """One column of the viewer grid: a token position and its ranked AO descriptions."""

    pos: int
    token: str  # the decoded token string at this position
    phrases: list[tuple[str, int]]  # (phrase, sample_count), most frequent first


def capture_residuals(
    backend: Any, rendered_ids: list[int], *, layer: int
) -> Float[Tensor, "pos d"]:
    """Layer-``L`` residual at every position of ``rendered_ids`` — one base-model forward.

    Identical mechanism to the training dump (``oracle_lens.dump.conversation_layers_resid``): an
    ``ActivationRecorder`` on ``backend.layers`` at ``layer``, so the captured ``h_p`` is exactly
    the distribution the AR/AO were fit on.
    """
    from oracle_lens.core.dump import conversation_layers_resid

    return conversation_layers_resid(backend, rendered_ids, layers=[layer])[layer]


def generate_phrases(
    ao_model: Any,
    tokenizer: Any,
    prompt: WVPrompt,
    vecs: Float[Tensor, "n d"],
    rep: Rep,
    *,
    alpha: float,
    scale: float,
    whitener: Any | None = None,
    k: int = 8,
    max_new: int = 24,
    temperature: float = 0.8,
    top_p: float = 0.95,
    micro_batch: int = 64,
    seed: int = 0,
) -> list[list[str]]:
    """For each of ``n`` injected vectors, sample ``k`` short AO descriptions -> ``[n][k]`` strings.

    Every (vector, sample) is one prompt with ``inject_rep(vec)`` replacing the slot embedding; the
    ``n * k`` prompts are generated in ``micro_batch``-sized chunks. Sampling (not greedy) so the
    ``k`` draws spread over the oracle's distribution for that position.
    """
    from oracle_lens.pipeline.scorer import prepare_reward_text

    device = next(ao_model.parameters()).device
    embed = ao_model.get_input_embeddings()
    inj = inject_rep(vecs.to(device), rep, alpha=alpha, scale=scale, whitener=whitener)
    n = inj.shape[0]
    # row order: vector 0 repeated k times, then vector 1, ... so a contiguous k-block is one column
    order = torch.repeat_interleave(torch.arange(n), k)
    base_ids = torch.tensor([prompt.input_ids], device=device)
    out: list[str] = []
    gen = torch.Generator(device="cpu").manual_seed(seed)
    for start in range(0, order.shape[0], micro_batch):
        rows = order[start : start + micro_batch]
        pe = embed(base_ids).expand(rows.shape[0], -1, -1).clone()
        pe[:, prompt.slot, :] = inj[rows].to(pe.dtype)
        attn = torch.ones(pe.shape[0], pe.shape[1], dtype=torch.long, device=device)
        torch.manual_seed(int(torch.randint(0, 2**31 - 1, (1,), generator=gen).item()))
        with torch.no_grad():
            g = ao_model.generate(
                inputs_embeds=pe,
                attention_mask=attn,
                max_new_tokens=max_new,
                do_sample=True,
                temperature=temperature,
                top_p=top_p,
            )
        out.extend(
            prepare_reward_text(t) for t in tokenizer.batch_decode(g, skip_special_tokens=True)
        )
    return [out[i * k : (i + 1) * k] for i in range(n)]


def aggregate_topk(samples: list[str], *, top: int = 8) -> list[tuple[str, int]]:
    """Dedup + count ``k`` raw samples into the ``top`` most frequent distinct phrases.

    Ties broken by first appearance (``Counter.most_common`` is stable in insertion order). Empty /
    whitespace-only samples are dropped.
    """
    cleaned = [s.strip() for s in samples if s.strip()]
    return Counter(cleaned).most_common(top)


def readout_text(
    backend: Any,
    ao_model: Any,
    tokenizer: Any,
    prompt: WVPrompt,
    text: str,
    rep: Rep,
    *,
    layer: int,
    alpha: float,
    scale: float,
    whitener: Any | None = None,
    k: int = 8,
    top: int = 8,
    max_new: int = 24,
    temperature: float = 0.8,
    add_special_tokens: bool = True,
    max_positions: int = 128,
) -> list[PositionReadout]:
    """Full per-position readout of ``text``: capture -> inject -> sample -> aggregate columns."""
    ids = tokenizer(text, add_special_tokens=add_special_tokens)["input_ids"][:max_positions]
    resid = capture_residuals(backend, ids, layer=layer)[: len(ids)]
    samples = generate_phrases(
        ao_model,
        tokenizer,
        prompt,
        resid,
        rep,
        alpha=alpha,
        scale=scale,
        whitener=whitener,
        k=k,
        max_new=max_new,
        temperature=temperature,
    )
    toks = tokenizer.convert_ids_to_tokens(ids)
    return [
        PositionReadout(
            pos=i, token=_clean_tok(toks[i]), phrases=aggregate_topk(samples[i], top=top)
        )
        for i in range(len(ids))
    ]


def _clean_tok(tok: str) -> str:
    """Render a BPE token for display (the ``Ġ``/``▁`` space marker -> a visible space)."""
    return tok.replace("Ġ", " ").replace("▁", " ")
