from dataclasses import dataclass
from contextlib import nullcontext

import torch
import torch.nn.functional as F
from torch import Tensor
from torch.nn.attention import SDPBackend, flex_attention, sdpa_kernel


@dataclass(frozen=True)
class BlockMaskInfo:
    q_len: int
    kv_len: int
    causal: bool
    sliding_window: int | None


# needs to be a list to avoid breaking torch compile -
# torch will someimtes use symints for the q/k len,
# which isn't hashable
BLOCK_MASK_CACHE: list[tuple[BlockMaskInfo, object]] = []
MAX_CACHE_SIZE = 8  # Maximum number of masks to cache


def make_causal_mask(q_len: int, kv_len: int):
    assert q_len <= kv_len
    q_offset = kv_len - q_len

    def causal_mask(b, h, q_idx, kv_idx):
        true_q_idx = q_idx + q_offset
        return true_q_idx >= kv_idx

    return causal_mask


def make_sliding_window_mask_causal(q_len: int, kv_len: int, sliding_window: int):
    assert q_len <= kv_len
    q_offset = kv_len - q_len

    def sliding_window_causal(b, h, q_idx, kv_idx):
        true_q_idx = q_idx + q_offset
        causal_mask = true_q_idx >= kv_idx
        window_mask = abs(true_q_idx - kv_idx) <= sliding_window
        return causal_mask & window_mask

    return sliding_window_causal


def make_sliding_window_mask_non_causal(q_len: int, kv_len: int, sliding_window: int):
    assert q_len <= kv_len
    q_offset = kv_len - q_len

    def sliding_window_non_causal(b, h, q_idx, kv_idx):
        true_q_idx = q_idx + q_offset
        window_mask = abs(true_q_idx - kv_idx) <= sliding_window
        return window_mask

    return sliding_window_non_causal


compiled_create_block_mask = torch.compile(flex_attention.create_block_mask)


def get_or_make_block_mask(
    q_len: int,
    kv_len: int,
    causal: bool,
    sliding_window: int | None,
    device: torch.device,
    cache_block_mask: bool,
):
    if not causal and sliding_window is None:
        return None

    info = BlockMaskInfo(
        q_len=q_len, kv_len=kv_len, causal=causal, sliding_window=sliding_window
    )

    # Linear scan through the cache list to find matching mask
    for i, (cached_info, cached_mask) in enumerate(BLOCK_MASK_CACHE):
        if cached_info == info:
            # Move to end (most recently used)
            BLOCK_MASK_CACHE.pop(i)
            BLOCK_MASK_CACHE.append((cached_info, cached_mask))
            return cached_mask

    # Not found in cache, create new mask
    if causal:
        if sliding_window is not None:
            mask_fn = make_sliding_window_mask_causal(q_len, kv_len, sliding_window)
        else:
            mask_fn = make_causal_mask(q_len, kv_len)

    else:
        mask_fn = make_sliding_window_mask_non_causal(q_len, kv_len, sliding_window)

    mask = compiled_create_block_mask(
        mask_fn,
        B=None,
        H=None,
        Q_LEN=q_len,
        KV_LEN=kv_len,
        device=device,
    )

    if cache_block_mask:
        # Add to end of cache
        BLOCK_MASK_CACHE.append((info, mask))

        # Remove oldest if cache is too large
        if len(BLOCK_MASK_CACHE) > MAX_CACHE_SIZE:
            BLOCK_MASK_CACHE.pop(0)

    return mask


@torch.compile
def compiled_flex_attention_forward(
    query: Tensor,
    key: Tensor,
    value: Tensor,
    block_mask: object,
) -> Tensor:
    return flex_attention.flex_attention(
        query, key, value, block_mask=block_mask, enable_gqa=True
    )


def flex_attention_forward(
    query: Tensor,
    key: Tensor,
    value: Tensor,
    causal: bool = True,
    sliding_window: int | None = None,
    cache_block_mask: bool = True,
) -> Tensor:
    block_mask = get_or_make_block_mask(
        q_len=query.shape[2],
        kv_len=key.shape[2],
        causal=causal,
        sliding_window=sliding_window,
        device=query.device,
        cache_block_mask=cache_block_mask,
    )

    return compiled_flex_attention_forward(query, key, value, block_mask)


def attention(
    query: Tensor,
    key: Tensor,
    value: Tensor,
    causal: bool = True,
    sliding_window: int | None = None,
    cache_block_mask: bool = True,
    provider: str | None = None,
) -> Tensor:
    """
    Q,K,V shape: [batch_size, num_heads, seq_len, head_dim]
    """

    device = query.device

    properties = torch.cuda.get_device_properties(device)
    is_amd = "AMD" in properties.name

    q_len = query.shape[-2]
    kv_len = key.shape[-2]

    q_dtype = query.dtype
    k_dtype = key.dtype
    v_dtype = value.dtype

    assert q_dtype == k_dtype == v_dtype, f"q_dtype: {q_dtype}, k_dtype: {k_dtype}, v_dtype: {v_dtype}"

    if provider is None:
        use_flex_attention = sliding_window is not None

        # the torch.sdpa causal mask does not do what we want with q_len != kv_len
        provider = (
            "flex" if use_flex_attention or (causal and q_len != kv_len) else "sdpa"
        )

    def sdpa_forward(backend):
        assert sliding_window is None

        ctx = nullcontext() if backend is None else sdpa_kernel([backend])

        with ctx:
            return F.scaled_dot_product_attention(
                query,
                key,
                value,
                is_causal=causal,
                enable_gqa=True,
            )

    match provider:
        case "flex":
            return flex_attention_forward(
                query=query,
                key=key,
                value=value,
                causal=causal,
                sliding_window=sliding_window,
                cache_block_mask=cache_block_mask,
            )
        case "sdpa":
            if is_amd:
                return sdpa_forward(SDPBackend.FLASH_ATTENTION)
            elif q_dtype == torch.float32:
                return sdpa_forward(None)
            else:
                # CUDNN is much faster than flash on hopper GPUs.
                return sdpa_forward(SDPBackend.CUDNN_ATTENTION)

        case "sdpa_flash":
            return sdpa_forward(SDPBackend.FLASH_ATTENTION)
        case "sdpa_cudnn":
            return sdpa_forward(SDPBackend.CUDNN_ATTENTION)
        case _:
            raise ValueError(f"Invalid provider: {provider}")
