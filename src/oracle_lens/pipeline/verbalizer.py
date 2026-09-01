"""Workspace-verbalizer prompt + injection utilities (official-NLA conventions).

Follows kitft/natural_language_autoencoders exactly where we have no reason to differ:
- The injection slot is a SINGLE-TOKEN CJK enclosed ideograph (U+3200-U+33FF), auto-picked per
  tokenizer, wrapped as ``<concept>CHAR</concept>`` with left/right-neighbor verification (the
  robust version of the "@"-merge problem).
- Injection is REPLACEMENT of that token's embedding with ``alpha * h / ||h||_2`` (raw
  activation, unit-L2, scaled large; the paper: "insert it in place of the special token's
  embedding").
- The prompt is chat-templated with ``enable_thinking=False`` (Qwen3.6 emits <think> otherwise).

Responses are expected to wrap explanations in ``<explanation>...</explanation>`` (the stock
schema); extraction/stripping for the reward path lives in ``scorer.prepare_reward_text``.
"""

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import torch
from jaxtyping import Float, Int
from torch import Tensor

# Stock NLA actor template (stage3_build.py defaults), trimmed to essentials.
WV_PROMPT_TEMPLATE = (
    "You are a meticulous AI researcher conducting an investigation into the internal "
    "representations of a language model. We will pass an activation vector from layer "
    "{layer} of the model into your context, enclosed in concept tags: "
    "<concept>{char}</concept>. Produce a short explanation of the content encoded by this "
    "activation, enclosed within <explanation> tags."
)

# Continuation variant (GT-continuation lens): the SAME scaffolding/slot, only the final
# instruction sentence differs — predict the text that FOLLOWED the read position (the span
# convention already makes the span the continuation; see gt_continuation_lens/design.md §4).
WV_CONTINUATION_PROMPT_TEMPLATE = (
    "You are a meticulous AI researcher conducting an investigation into the internal "
    "representations of a language model. We will pass an activation vector from layer "
    "{layer} of the model into your context, enclosed in concept tags: "
    "<concept>{char}</concept>. Produce the text that immediately follows this activation in "
    "the model's generation, enclosed within <explanation> tags."
)

# Minimal continuation variant (AO ladder, user-approved 2026-07-29): same slot mechanism and
# <explanation> target contract, ~28 rendered tokens vs ~82 — the verbose persona preamble is
# pure FLOPs (72% of every training sequence was prompt re-encoding at the GT wording).
WV_CONTINUATION_MIN_PROMPT_TEMPLATE = (
    "An activation vector from layer {layer} of a language model is enclosed in concept tags: "
    "<concept>{char}</concept>. Produce the text that immediately follows this activation, in "
    "<explanation> tags."
)

# Raw continuation variant (u64 scaling run, Agam 2026-08-08): NO <explanation> tags anywhere —
# the target is the raw span + EOS, and the enclosure is renamed to <activation> tags (Agam's
# wording, locked). Same slot mechanism; pair with AOLadderDataset(wrap_tags=False).
WV_CONTINUATION_RAW_PROMPT_TEMPLATE = (
    "An activation vector from layer {layer} of a language model is enclosed in activation "
    "tags: <activation>{char}</activation>. Produce the text that immediately follows this "
    "activation."
)

# Tag-free "explains" framing (register-SFT experiment, Agam 2026-08-11): the plain explain
# instruction with NO tags anywhere and NO extra constraint clauses (Agam) — <activation>
# enclosure like continuation_raw, raw-span targets (wrap_tags=False). Tests whether prompt
# wording alone shifts the u64 AO's register away from pure document-continuation.
WV_EXPLAIN_RAW_PROMPT_TEMPLATE = (
    "An activation vector from layer {layer} of a language model is enclosed in activation "
    "tags: <activation>{char}</activation>. Produce the text that explains what this "
    "activation encodes."
)

# Tag-free multi-concept bullet prompt (distill round 1, Agam 2026-08-11): the student reads a
# GT activation and emits its selected NNOMP picks as '- ' bullets, shortest-first, one per
# line, EOS-terminated — no count in the wording (Agam: do NOT say "4 distinct"), no tags.
# Pair with distill_v1 targets whose bullet text is stored verbatim (prompt_mode
# "concepts_raw"); parseability is guaranteed by the assembler, which drops sets where a pick
# contains a line starting with "- ".
WV_CONCEPTS_RAW_PROMPT_TEMPLATE = (
    "An activation vector from layer {layer} of a language model is enclosed in activation "
    "tags: <activation>{char}</activation>. Produce distinct concepts that encode this "
    "activation, each as a '- ' bullet on its own line."
)

# "Explains" framing (AO-ladder prompt A/B, user 2026-07-29): same slot + <explanation> target,
# but frames the task as describing the activation's content rather than continuing from it. The
# label is unchanged (the exact span), so this tests whether the framing helps or fights the task.
WV_EXPLAIN_PROMPT_TEMPLATE = (
    "An activation vector from layer {layer} of a language model is enclosed in concept tags: "
    "<concept>{char}</concept>. Produce the text that explains what this activation encodes, in "
    "<explanation> tags."
)

# Minimal-graft desc variant (Camila 2026-08-10, warmstart round 2): the trained explain
# skeleton VERBATIM with exactly one constraint clause inserted — the only delta vs the
# training prompt is the name-the-content / anti-continuation instruction. Designed after the
# 250-token precise prompt proved too OOD for the step3002 AO (degen echo; hit-rate drop,
# warmstart_system_prompt_exploration.md finding 1) — this isolates the instruction while
# staying at ~45 rendered tokens, near the ~28-token training distribution.
WV_EXPLAIN_SPECIFIC_PROMPT_TEMPLATE = (
    "An activation vector from layer {layer} of a language model is enclosed in concept tags: "
    "<concept>{char}</concept>. Produce the text that explains what this activation encodes — "
    "name the specific latent content the model is representing; do not continue or paraphrase "
    "the surrounding text — in <explanation> tags."
)

# DESC "precise" few-shot prompt (Camila 2026-08-04, r2sf desc-eliciting round; verbatim copy of
# _COMMON+_PRECISE in scripts/ola/olens_r2sf_desc_prompt_pilot.py). Same {layer} /
# <concept>{char}</concept> slot / <explanation> primitives as every WV template, plus the
# anti-continuation + specific-referent constraints and 3 worked few-shots. Used by the warm-start
# system-prompt exploration to test whether this prompt (vs the short explain prompt) changes what
# the AO surfaces (warmstart_system_prompt_exploration.md).
WV_PRECISE_PROMPT_TEMPLATE = (
    "You are a meticulous AI researcher conducting an investigation into the internal "
    "representations of a language model. We will pass an activation vector from layer "
    "{layer} of the model into your context, enclosed in concept tags: "
    "<concept>{char}</concept>. "
    "Describe, precisely and succinctly but in complete detail, what the model is thinking "
    "about at this exact position and layer. Name the SPECIFIC latent content — the actual "
    "value being computed, the specific referent, or the model's own strategic stance — not a "
    "vague category, and never invent content the activation does not support. Do NOT continue "
    "or paraphrase the text; state what the model is representing.\n\n"
    "Examples of the KIND of description wanted:\n"
    '- For "((a-b)*c)+d=" at the "=": in earlier-middle layers the model is holding the '
    "value of (a-b); in later-middle layers the value of (a-b)*c; only in very late layers the "
    "final answer. Describe the intermediate appropriate to THIS layer.\n"
    "- For a passage where a widow pours out her late husband's daily coffee on the anniversary "
    'of his death and finally cries: "grief at the loss of a spouse" — the specific relation, '
    'not merely "grief" and not a fabricated variant like "grief at the loss of a child".\n'
    "- For a harmful multiple-choice question where the assistant deliberately answers with the "
    'wrong option: "the model is sandbagging for safety, intentionally choosing the incorrect '
    'answer to avoid causing harm".\n\n'
    "Produce 4 distinct such descriptions, each about 32 tokens long, each enclosed within its "
    "own <explanation> tags."
)


# List variant (self-distillation student): the SAME injection slot, but ask for k distinct
# explanations, each in its own <explanation> block — the multiple-outputs-in-one-forward objective.
WV_LIST_PROMPT_TEMPLATE = (
    "You are a meticulous AI researcher conducting an investigation into the internal "
    "representations of a language model. We will pass an activation vector from layer "
    "{layer} of the model into your context, enclosed in concept tags: "
    "<concept>{char}</concept>. Produce {k} distinct short explanations of the content encoded by "
    "this activation, each enclosed within its own <explanation> tags."
)

# List variant with a stated per-phrase token length (the round-2 students). The length wording
# is DESCRIPTIVE only — n is enforced by max_tokens at sampling/eval time and by token-level
# truncation when the training targets are assembled (user decision 2026-07-29), never by
# trusting the model to count.
WV_LIST_N_PROMPT_TEMPLATE = (
    "You are a meticulous AI researcher conducting an investigation into the internal "
    "representations of a language model. We will pass an activation vector from layer "
    "{layer} of the model into your context, enclosed in concept tags: "
    "<concept>{char}</concept>. Produce {k} distinct explanations of the content encoded by "
    "this activation, each about {n} tokens long, each enclosed within its own "
    "<explanation> tags."
)

# "Up to k" list variant (the r2sf filtered-selection round, Camila 2026-08-03): same as the
# k/n list prompt but the student may stop early — variable-k targets come from the dedup (F2)
# and deflation-stop (F4) filters, and the wording licenses emitting only as many phrases as
# the activation needs. k/n stay FIXED per run (constant prompt length; see render docstring).
WV_LIST_UPTO_N_PROMPT_TEMPLATE = (
    "You are a meticulous AI researcher conducting an investigation into the internal "
    "representations of a language model. We will pass an activation vector from layer "
    "{layer} of the model into your context, enclosed in concept tags: "
    "<concept>{char}</concept>. Produce up to {k} distinct explanations of the content encoded "
    "by this activation — only as many as are needed to cover its distinct content — each "
    "about {n} tokens long, each enclosed within its own <explanation> tags."
)

# Desc student variant (r2sf-dp round, Camila 2026-08-04): the student is trained on targets
# sampled from the teacher under the DESC-eliciting "precise" prompt (86.6% DESC in the pilot,
# r2sf_filter_plan.md), so its own system prompt asks for latent-content DESCRIPTIONS in the
# same terms. Deliberately example-free — the teacher needed the three worked few-shots to
# GENERATE desc-style targets; the student learns the style from the targets themselves, and a
# shorter fixed prompt keeps the trainer's single-prompt harness cheap.
WV_LIST_UPTO_N_DESC_PROMPT_TEMPLATE = (
    "You are a meticulous AI researcher conducting an investigation into the internal "
    "representations of a language model. We will pass an activation vector from layer "
    "{layer} of the model into your context, enclosed in concept tags: "
    "<concept>{char}</concept>. Describe, precisely and succinctly but in complete detail, "
    "what the model is thinking about at this exact position and layer. Name the SPECIFIC "
    "latent content — the actual value being computed, the specific referent, or the model's "
    "own strategic stance — not a vague category, and never invent content the activation "
    "does not support. Do NOT continue or paraphrase the text; state what the model is "
    "representing. Produce up to {k} distinct such descriptions — only as many as are needed "
    "to cover the activation's distinct content — each about {n} tokens long, each enclosed "
    "within its own <explanation> tags."
)


@dataclass(frozen=True)
class WVPrompt:
    """A rendered WV prompt with the injection slot located from the ACTUAL tokenization."""

    input_ids: list[int]
    slot: int  # index of the injection token within input_ids
    char: str
    char_id: int


def _render_with_slot(tokenizer: Any, make_text: Any) -> WVPrompt:
    """Pick the injection char by verifying it survives as ONE token in the FULL rendered prompt.

    A char can tokenize to one id in isolation yet merge with its neighbors inside the real
    chat-templated prompt (BPE is context-sensitive — this bit us with the first candidate),
    so candidacy is judged on the final token sequence, nothing less. ``make_text(char)`` builds
    the user message for a candidate char (the injection tag position is fixed; text after it —
    e.g. the requested count k — does not affect slot detection, which scans for ``char_id``).
    """
    for code in range(0x3200, 0x3400):
        char = chr(code)
        ids = tokenizer(char, add_special_tokens=False)["input_ids"]
        if len(ids) != 1:
            continue
        char_id = ids[0]
        messages = [{"role": "user", "content": make_text(char)}]
        out = tokenizer.apply_chat_template(
            messages, tokenize=True, add_generation_prompt=True, enable_thinking=False
        )
        # BatchEncoding iterates its KEYS, not ids — index explicitly (cost us a zero-slot bug)
        rendered: list[int] = out if isinstance(out, list) else out["input_ids"]
        slots = [i for i, t in enumerate(rendered) if t == char_id]
        if len(slots) == 1:
            return WVPrompt(input_ids=rendered, slot=slots[0], char=char, char_id=char_id)
    raise ValueError("no enclosed ideograph survives as a single token in the full prompt")


def render_wv_prompt(tokenizer: Any, *, layer: int = 44) -> WVPrompt:
    """The single-explanation WV prompt (the shipped AO's actor prompt)."""
    return _render_with_slot(
        tokenizer, lambda char: WV_PROMPT_TEMPLATE.format(layer=layer, char=char)
    )


def render_wv_continuation_prompt(tokenizer: Any, *, layer: int) -> WVPrompt:
    """The GT-continuation prompt (one render per layer; the layer number is in the text)."""
    return _render_with_slot(
        tokenizer, lambda char: WV_CONTINUATION_PROMPT_TEMPLATE.format(layer=layer, char=char)
    )


def render_wv_continuation_min_prompt(tokenizer: Any, *, layer: int) -> WVPrompt:
    """The minimal continuation prompt (AO ladder; one render per layer)."""
    return _render_with_slot(
        tokenizer, lambda char: WV_CONTINUATION_MIN_PROMPT_TEMPLATE.format(layer=layer, char=char)
    )


def render_wv_explain_prompt(tokenizer: Any, *, layer: int) -> WVPrompt:
    """The 'explains' framing (AO-ladder A/B); one render per layer."""
    return _render_with_slot(
        tokenizer, lambda char: WV_EXPLAIN_PROMPT_TEMPLATE.format(layer=layer, char=char)
    )


def render_wv_continuation_raw_prompt(tokenizer: Any, *, layer: int) -> WVPrompt:
    """The tag-free continuation prompt (u64 run); target = raw span, no <explanation> wrapper."""
    return _render_with_slot(
        tokenizer, lambda char: WV_CONTINUATION_RAW_PROMPT_TEMPLATE.format(layer=layer, char=char)
    )


def render_wv_explain_raw_prompt(tokenizer: Any, *, layer: int) -> WVPrompt:
    """The tag-free explain prompt (register-SFT experiment); raw-span targets, no tags."""
    return _render_with_slot(
        tokenizer, lambda char: WV_EXPLAIN_RAW_PROMPT_TEMPLATE.format(layer=layer, char=char)
    )


def render_wv_concepts_raw_prompt(tokenizer: Any, *, layer: int) -> WVPrompt:
    """The tag-free bullet-list prompt (distill round 1); targets are '- ' bullet blocks."""
    return _render_with_slot(
        tokenizer, lambda char: WV_CONCEPTS_RAW_PROMPT_TEMPLATE.format(layer=layer, char=char)
    )


def render_wv_explain_specific_prompt(tokenizer: Any, *, layer: int) -> WVPrompt:
    """The minimal-graft explain variant (explain + one desc clause); one render per layer."""
    return _render_with_slot(
        tokenizer, lambda char: WV_EXPLAIN_SPECIFIC_PROMPT_TEMPLATE.format(layer=layer, char=char)
    )


def render_wv_precise_prompt(tokenizer: Any, *, layer: int) -> WVPrompt:
    """The DESC 'precise' few-shot prompt (one render per layer)."""
    return _render_with_slot(
        tokenizer, lambda char: WV_PRECISE_PROMPT_TEMPLATE.format(layer=layer, char=char)
    )


# Every fixed-arity WV renderer by canonical name, plus the training-meta aliases ("gt" was
# the original name of the continuation prompt, "min" of the minimal variant — ao_runs
# meta.json records those). render_wv_list_prompt is excluded (it needs k/n). The naming
# trap this registry kills: prompt_kind="explanation" (the verbose NLA persona prompt) is
# NOT the AO ladder's prompt="explain" (WV_EXPLAIN_PROMPT_TEMPLATE) — dispatching by two
# hand-rolled if/else chains silently ran AOs on the wrong template AND the wrong task.
PROMPT_RENDERERS: dict[str, Callable[..., WVPrompt]] = {
    "continuation": render_wv_continuation_prompt,
    "continuation_min": render_wv_continuation_min_prompt,
    "continuation_raw": render_wv_continuation_raw_prompt,
    "explain_raw": render_wv_explain_raw_prompt,
    "concepts_raw": render_wv_concepts_raw_prompt,
    "explanation": render_wv_prompt,
    "explain": render_wv_explain_prompt,
    "explain_specific": render_wv_explain_specific_prompt,
    "precise": render_wv_precise_prompt,
}
PROMPT_ALIASES = {"gt": "continuation", "min": "continuation_min"}


def renderer_for(kind: str) -> Callable[..., WVPrompt]:
    """The WV renderer for a prompt-kind name or training-meta alias."""
    key = PROMPT_ALIASES.get(kind, kind)
    if key not in PROMPT_RENDERERS:
        raise ValueError(
            f"unknown prompt kind {kind!r} — have {sorted(PROMPT_RENDERERS)} "
            f"(+ aliases {sorted(PROMPT_ALIASES)})"
        )
    return PROMPT_RENDERERS[key]


def render_wv_list_prompt(
    tokenizer: Any, *, layer: int = 44, k: int, n: int = 0, upto: bool = False, desc: bool = False
) -> WVPrompt:
    """The list WV prompt: ask for ``k`` distinct explanations (the self-distillation student).

    Same injection slot as ``render_wv_prompt``; ``k``/``n`` appear after the concept tag so they
    never perturb slot detection. ``n > 0`` uses the length-stating template (round-2 students;
    wording only — n is enforced by max_tokens/truncation, not the prompt). ``upto=True`` uses
    the "up to k" wording (r2sf variable-k students; requires ``n > 0``). ``desc=True`` uses the
    desc-student template (r2sf-dp round: up-to-k latent-content descriptions; implies the
    up-to-k semantics and requires ``n > 0``; ``upto`` is ignored). Render with FIXED
    ``k``/``n`` per training run to keep the prompt length constant (the trainer's single-prompt
    harness assumes a fixed prompt); per-example k needs the per-example-prompt path.
    """
    if desc:
        if n <= 0:
            raise ValueError("desc=True requires n > 0 (the desc template states n)")
        return _render_with_slot(
            tokenizer,
            lambda char: WV_LIST_UPTO_N_DESC_PROMPT_TEMPLATE.format(
                layer=layer, char=char, k=k, n=n
            ),
        )
    if upto:
        if n <= 0:
            raise ValueError("upto=True requires n > 0 (the r2sf up-to-k template states n)")
        return _render_with_slot(
            tokenizer,
            lambda char: WV_LIST_UPTO_N_PROMPT_TEMPLATE.format(layer=layer, char=char, k=k, n=n),
        )
    if n > 0:
        return _render_with_slot(
            tokenizer,
            lambda char: WV_LIST_N_PROMPT_TEMPLATE.format(layer=layer, char=char, k=k, n=n),
        )
    return _render_with_slot(
        tokenizer, lambda char: WV_LIST_PROMPT_TEMPLATE.format(layer=layer, char=char, k=k)
    )


def render_wv_list4_prompt(tokenizer: Any, *, layer: int = 44) -> WVPrompt:
    """List student prompt with FIXED k=4, n=32 (the r2s m*_64 students; WV_LIST_N template).

    Fixed-arity so it fits the ``renderer_for`` registry (which calls ``renderer(tok, layer=L)``).
    """
    return render_wv_list_prompt(tokenizer, layer=layer, k=4, n=32)


PROMPT_RENDERERS["list4"] = render_wv_list4_prompt


def inject_embeds(
    embed_layer: Any,
    prompt: WVPrompt,
    acts: Float[Tensor, "b d"],
    *,
    alpha: float,
) -> tuple[Float[Tensor, "b s d"], Int[Tensor, "b s"]]:
    """Build [b, seq, d] prompt embeddings with ``alpha * h/||h||`` REPLACING the slot embed.

    WARNING: this path ALWAYS unit-normalizes — it has no ``transform`` parameter, so any
    caller wanting ``raw``/``scaled`` injection semantics must not route through here (the
    eval pipeline uses ``olens_sglang.common.inject_vecs`` instead; a readout/visualizer
    that calls this silently ignores a raw/scaled request).
    """
    device = acts.device
    ids = torch.tensor([prompt.input_ids], dtype=torch.long, device=device)
    base: Tensor = embed_layer(ids)  # [1, s, d]
    b = acts.shape[0]
    embeds = base.expand(b, -1, -1).clone()
    unit = acts / acts.norm(dim=-1, keepdim=True).clamp_min(1e-9)
    embeds[:, prompt.slot, :] = (alpha * unit).to(embeds.dtype)
    attn = torch.ones(b, ids.shape[1], dtype=torch.long, device=device)
    return embeds, attn
