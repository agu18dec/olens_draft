"""Shared GPU-test fixtures: a CUDA skip guard and a once-per-session cuda backend.

GPU tests are also tagged ``@pytest.mark.gpu`` and deselected by default (see
``addopts`` in pyproject.toml); run them with ``uv run pytest -m gpu``.
"""


import pytest
import torch

from oracle_lens.model import ModelBackend

# The smallest public model the production scripts default to — a ~1 GB download.
_SMOKE_MODEL = "Qwen/Qwen2.5-0.5B-Instruct"


@pytest.fixture(scope="session")
def backend() -> ModelBackend:
    """Load the 0.5B model on cuda once and share it across the GPU tests."""
    return ModelBackend(_SMOKE_MODEL, device="cuda", dtype=torch.bfloat16)


@pytest.fixture(autouse=True)
def _cpu_torch_qwen3_5_kernels(request: pytest.FixtureRequest) -> None:
    """Mask the CUDA-only qwen3_5 fast-path kernels for CPU tests (everything not ``-m gpu``).

    The AO/AR trainers require flash-attn + fla + causal-conv1d in this venv, but
    ``modeling_qwen3_5`` binds those CUDA/Triton kernels at import and layers capture them at
    ``__init__`` — so the suite's CPU tiny-model forwards crash (``Expected x.is_cuda()`` /
    Triton "cpu tensor?"). Masking the five module-level names restores the pure-torch
    fallbacks, i.e. exactly the environment the suite was green in before the kernels were
    installed. GPU-marked tests keep the real kernels.
    """
    if request.node.get_closest_marker("gpu") is not None:
        return
    try:
        from transformers.models.qwen3_5 import modeling_qwen3_5 as m
    except ImportError:  # pragma: no cover
        return
    mp = pytest.MonkeyPatch()
    request.addfinalizer(mp.undo)
    for name in (
        "causal_conv1d_fn",
        "causal_conv1d_update",
        "chunk_gated_delta_rule",
        "fused_recurrent_gated_delta_rule",
        "FusedRMSNormGated",
    ):
        mp.setattr(m, name, None, raising=False)
