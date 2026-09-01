"""Model access: ``ModelBackend``, a ``jlens.protocol.LensModel`` with the extras we need.

The official adapter (``jlens.hf.from_hf``) supplies the model-layout detection and the
protocol surface (``encode`` / ``forward`` / ``unembed`` / ``layers``), so a backend can be
passed directly to ``jlens.fit`` / ``JacobianLens.apply``. On top of that this class keeps
what the reference implementation leaves out and our experiments depend on:

- ``run``: one locked forward pass returning the full ``[layer, pos, d_model]`` residual
  stack plus final logits (the batched-readout entry point);
- ``generate_texts`` + chat templating (``enable_thinking`` handling for reasoning models);
- raw ``W_U`` / ``final_norm`` pieces consumed by the batched lens math in
  :mod:`oracle_lens.jlens_readout`;
- ``_model_lock``: all model access is serialized — forward hooks register on the *shared*
  module list, so two concurrent requests (e.g. an ASGI server with ``max_inputs>1``) would
  cross-contaminate each other's captured residuals.
"""

import functools
import threading
from collections.abc import Sequence
from typing import Any

import torch
from jaxtyping import Float
from torch import Tensor
from transformers import AutoModelForCausalLM, AutoTokenizer

from jlens.hf import HFLensModel, from_hf
from jlens.hooks import ActivationRecorder


def load_causal_lm(
    model_id: str,
    *,
    dtype: torch.dtype = torch.float32,
    device: str = "cpu",
    attn_implementation: str = "sdpa",
    trust_remote_code: bool = False,
    disable_cache: bool = False,
    device_map: str | None = None,
    experts_impl: str | None = None,
) -> Any:
    """A frozen, eval-mode causal LM on ``device`` — the shared prep all load sites need.

    ``disable_cache`` sets ``config.use_cache=False`` (the Jacobian graph needs it; KV-cache
    paths can detach the autograd graph). Returns the model only; callers that also need a
    tokenizer load it themselves (a one-liner, not the part that was duplicated).

    ``device_map`` (e.g. ``"auto"``) hands placement to accelerate for models too large for a
    single GPU: it is passed straight to ``from_pretrained`` and the manual ``model.to(device)``
    is skipped (accelerate already sharded the weights across devices). When ``None`` (default)
    behaviour is unchanged — single-device load then ``.to(device)``.

    ``experts_impl`` (e.g. ``"eager"``) overrides the fused-MoE expert dispatch. The stock
    ``grouped_mm`` kernel is a torch custom op whose fake/meta impl trips a
    ``Tensor on device meta`` error under ``device_map`` sharding + a retained autograd graph;
    forcing the plain eager expert loop (value-identical, same gradients) sidesteps it. The
    integrations wrapper reads ``self.config._experts_implementation`` per forward, so setting
    it post-load on each experts module's config takes effect immediately.
    """
    model: Any = AutoModelForCausalLM.from_pretrained(
        model_id,
        dtype=dtype,
        attn_implementation=attn_implementation,
        trust_remote_code=trust_remote_code,
        **({"device_map": device_map} if device_map is not None else {}),
    )
    if device_map is None:
        model.to(device)
    if experts_impl is not None:
        n_set = 0
        for module in model.modules():
            if hasattr(module, "gate_up_proj") and hasattr(module, "config"):
                module.config._experts_implementation = experts_impl
                n_set += 1
        print(f"[load_causal_lm] _experts_implementation={experts_impl!r} on {n_set} experts")
    model.eval()
    model.requires_grad_(False)
    if disable_cache:
        model.config.use_cache = False
    return model


def generation_kwargs(
    *,
    do_sample: bool,
    temperature: float,
    top_p: float,
    banned_token_ids: Sequence[int] | None,
) -> dict[str, Any]:
    """Build the ``model.generate`` sampling kwargs (pure, so it is unit-testable model-free).

    Greedy decoding ignores ``temperature``/``top_p``; ``banned_token_ids`` map to HF's
    ``suppress_tokens`` (-inf at every decode step).
    """
    kwargs: dict[str, Any] = (
        {"do_sample": True, "temperature": temperature, "top_p": top_p}
        if do_sample
        else {"do_sample": False}
    )
    if banned_token_ids:
        kwargs["suppress_tokens"] = list(banned_token_ids)
    return kwargs


class ModelBackend:
    """Loads a HF causal LM and exposes residuals, W_U, and the final RMSNorm.

    Satisfies ``jlens.protocol.LensModel`` (via the official HF adapter), so it can be handed
    straight to ``jlens.fit`` or ``JacobianLens.apply``.
    """

    def __init__(
        self,
        model_id: str,
        device: str = "cpu",
        dtype: torch.dtype = torch.float32,
        attn_implementation: str = "sdpa",
        device_map: str | None = None,
        experts_impl: str | None = None,
    ) -> None:
        self.model_id = model_id
        self.device = device
        self.model: Any = load_causal_lm(
            model_id,
            dtype=dtype,
            device=device,
            attn_implementation=attn_implementation,
            device_map=device_map,
            experts_impl=experts_impl,
        )
        self.tokenizer: Any = AutoTokenizer.from_pretrained(model_id)
        # force_bos=False: the official default flips add_bos_token on instruct checkpoints
        # (a fitting-quality concern); our probes and recorded experiments never prepend BOS,
        # and changing tokenization here would silently shift every readout position.
        self._hf: HFLensModel = from_hf(self.model, self.tokenizer, force_bos=False)
        self._model_lock = threading.Lock()

        cfg = self.model.config.get_text_config()
        self.n_layers: int = self._hf.n_layers
        self.d_model: int = self._hf.d_model
        self.d_vocab: int = cfg.vocab_size
        self.rms_eps: float = cfg.rms_norm_eps

        layout = self._hf.layout
        self.W_U: Float[Tensor, "d_vocab d_model"] = getattr(self.model, layout.lm_head).weight
        text_module = functools.reduce(getattr, layout.path.split("."), self.model)
        norm_module = getattr(text_module, layout.norm)
        norm_weight = norm_module.weight
        assert isinstance(norm_weight, Tensor)
        self.norm_weight: Float[Tensor, "d_model"] = norm_weight
        # Under accelerate sharding the embedding, lm_head, and final norm live on whatever
        # shard accelerate chose; ``device`` stays the single readout device, so pull the
        # (read-only) unembed pieces onto it and route inputs to the embedding's shard.
        self._input_device: torch.device = (
            self._hf.input_device if device_map is not None else torch.device(device)
        )
        if device_map is not None:
            self.W_U = self.W_U.to(device)
            self.norm_weight = self.norm_weight.to(device)
        # Gemma-family RMSNorm scales by (1 + weight), not weight (zero-centered init);
        # using the raw weight there per-channel-misscales every lens readout.
        self._norm_offset: float = 1.0 if type(norm_module).__name__.startswith("Gemma") else 0.0

    # --- jlens.protocol.LensModel surface (delegated to the official adapter) -------------

    @property
    def layers(self) -> Any:
        """The decoder blocks (``nn.ModuleList``) — the hook targets for capture and edits."""
        return self._hf.layers

    def encode(self, text: str, *, max_length: int = 512) -> Tensor:
        return self._hf.encode(text, max_length=max_length)

    def forward(self, input_ids: Tensor) -> Any:
        return self._hf.forward(input_ids)

    def unembed(self, residual: Tensor) -> Tensor:
        return self._hf.unembed(residual)

    # --- our extensions --------------------------------------------------------------------

    def run(
        self, text: str
    ) -> tuple[Float[Tensor, "layer pos d_model"], Float[Tensor, "pos d_vocab"]]:
        """Forward pass on one prompt; return per-layer residuals (resid_post) and final logits."""
        input_ids = self.tokenizer(text, return_tensors="pt").input_ids.to(self._input_device)
        with (
            self._model_lock,
            torch.no_grad(),
            ActivationRecorder(self.layers, at=range(self.n_layers)) as recorder,
        ):
            output = self.model(input_ids, use_cache=False)
            # .to(self.device): under device_map sharding each layer's capture lives on its
            # own shard, and a cross-device torch.stack raises.
            resid = torch.stack(
                [recorder.activations[i][0].to(self.device) for i in range(self.n_layers)], dim=0
            )  # [layer, pos, d_model]
        return resid, output.logits[0].to(self.device)

    def generate_texts(
        self,
        prompt: str,
        max_new_tokens: int = 8,
        do_sample: bool = False,
        temperature: float = 1.0,
        num_return_sequences: int = 1,
        *,
        top_p: float = 1.0,
        banned_token_ids: Sequence[int] | None = None,
        seed: int | None = None,
    ) -> list[str]:
        """Continuations for one prompt (greedy by default; sampled when ``do_sample``).

        Generation runs with the KV cache, so any active prefill edit (registered forward
        hooks that modify the prompt's residuals) propagates into every sampled token.

        ``top_p`` is the nucleus-sampling cutoff (only used when ``do_sample``).
        ``banned_token_ids`` are suppressed at *every* decode step (HF ``suppress_tokens``),
        so an injected direction can stay live while the model is forbidden from emitting
        those tokens. ``seed`` re-seeds the global RNG before generation, so the same ``seed``
        across conditions yields paired sampling.
        """
        if seed is not None:
            torch.manual_seed(seed)
        encoded = self.tokenizer(prompt, return_tensors="pt").to(self._input_device)
        sampling = generation_kwargs(
            do_sample=do_sample,
            temperature=temperature,
            top_p=top_p,
            banned_token_ids=banned_token_ids,
        )
        with self._model_lock, torch.no_grad():
            out = self.model.generate(
                **encoded,
                max_new_tokens=max_new_tokens,
                num_return_sequences=num_return_sequences,
                pad_token_id=self.tokenizer.pad_token_id or self.tokenizer.eos_token_id,
                **sampling,
            )
        new_tokens = out[:, encoded["input_ids"].shape[1] :]
        return [self.tokenizer.decode(seq, skip_special_tokens=True) for seq in new_tokens]

    def apply_chat_template(
        self,
        user_message: str,
        *,
        system: str | None = None,
        enable_thinking: bool | None = None,
    ) -> str:
        """Wrap ``user_message`` in the model's chat template (with the generation prompt).

        Returns the templated *string* (not ids), so the visualizer's existing string-based
        ``run``/``token_strs`` tokenize it exactly as the model sees it — the special role
        tokens (``<|im_start|>`` …) then appear as their own columns, which is where the
        lens readout for the about-to-speak Assistant slot lives.

        ``enable_thinking`` is forwarded to the template only when set (reasoning models like
        Qwen3 branch on it to emit/omit the empty ``<think></think>`` block); the default
        ``None`` leaves the call byte-identical to before, and a template that doesn't accept
        the kwarg falls back gracefully.
        """
        messages = [{"role": "system", "content": system}] if system else []
        messages.append({"role": "user", "content": user_message})
        kwargs: dict[str, Any] = {"tokenize": False, "add_generation_prompt": True}
        if enable_thinking is not None:
            kwargs["enable_thinking"] = enable_thinking
        try:
            templated = self.tokenizer.apply_chat_template(messages, **kwargs)
        except TypeError:  # template/signature doesn't accept enable_thinking
            kwargs.pop("enable_thinking", None)
            templated = self.tokenizer.apply_chat_template(messages, **kwargs)
        return str(templated)

    def apply_chat_template_messages(
        self,
        messages: list[dict[str, str]],
        *,
        add_generation_prompt: bool = False,
        enable_thinking: bool | None = None,
    ) -> str:
        """Template a **full multi-turn conversation** (a ``[{"role", "content"}, ...]`` list) into
        the model's chat format, preserving every turn's role markers (``<|im_start|>user`` …
        ``<|im_start|>assistant`` …).

        Unlike :meth:`apply_chat_template` (which wraps a single user turn and adds the generation
        prompt), this is for reading the lens over a long, realistically-formatted dialogue — e.g.
        SWE-chat ``chat_full`` mode, where the whole agent trajectory is the readout context. The
        caller truncates the returned string to ``t_max`` tokens. ``add_generation_prompt`` defaults
        to ``False`` (no about-to-speak slot — we are inspecting context, not generating).
        """
        kwargs: dict[str, Any] = {"tokenize": False, "add_generation_prompt": add_generation_prompt}
        if enable_thinking is not None:
            kwargs["enable_thinking"] = enable_thinking
        try:
            templated = self.tokenizer.apply_chat_template(messages, **kwargs)
        except TypeError:  # template/signature doesn't accept enable_thinking
            kwargs.pop("enable_thinking", None)
            templated = self.tokenizer.apply_chat_template(messages, **kwargs)
        return str(templated)

    def token_strs(self, text: str) -> list[str]:
        """Per-position token strings for the visualizer's column labels.

        Uses the same tokenizer ``run`` feeds the model, so the strings line up with the
        ``pos`` axis of the captured residuals. Decoded per-id (not ``convert_ids_to_tokens``)
        so multibyte scripts (e.g. Chinese) render as real characters, not byte-level pieces;
        special role tokens (``<|im_start|>``) still come through verbatim.
        """
        ids = self.tokenizer(text)["input_ids"]
        return [self.tokenizer.decode([i]) for i in ids]

    def final_norm(self, x: Float[Tensor, "... d_model"]) -> Float[Tensor, "... d_model"]:
        """The model's pre-unembed RMSNorm (the ``norm`` in softmax(W_U·norm(·)))."""
        variance = x.pow(2).mean(dim=-1, keepdim=True)
        return x * torch.rsqrt(variance + self.rms_eps) * (self.norm_weight + self._norm_offset)
