import pydra
import torch

from mytorch_lightning.utils.attention import attention


class TestConfig(pydra.Config):
    batch_size: int = 2
    num_heads: int = 8
    num_kv_heads: int | None = None
    seq_len: int = 512
    head_dim: int = 64
    causal: bool = True
    dtype: str = "bfloat16"  # Options: float32, float16, bfloat16
    device: str = "cuda:0"
    atol: float = 1e-5
    rtol: float = 1e-4
    seed: int = 42

    def finalize(self):
        super().finalize()
        if self.num_kv_heads is None:
            self.num_kv_heads = self.num_heads


def get_dtype(dtype_str: str):
    dtype_map = {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }
    return dtype_map[dtype_str]


def adiff(a: torch.Tensor, b: torch.Tensor) -> float:
    """Maximum absolute difference between tensors."""
    return torch.max(torch.abs(a - b)).item()


def rdiff(a: torch.Tensor, b: torch.Tensor, eps: float = 1e-8) -> float:
    """Mean relative difference between tensors using 2*|a-b|/(|a|+|b|+eps)."""
    numerator = 2 * torch.abs(a - b)
    denominator = torch.abs(a) + torch.abs(b) + eps
    return torch.mean(numerator / denominator).item()


def main(config: TestConfig):
    print(f"Testing attention with config: {config.to_dict()}")

    torch.manual_seed(config.seed)
    dtype = get_dtype(config.dtype)
    device = torch.device(config.device)

    # Create test tensors
    query_shape = (
        config.batch_size,
        config.num_heads,
        config.seq_len,
        config.head_dim,
    )
    kv_shape = (
        config.batch_size,
        config.num_kv_heads,
        config.seq_len,
        config.head_dim,
    )
    query = torch.randn(query_shape, dtype=dtype, device=device, requires_grad=True)
    key = torch.randn(kv_shape, dtype=dtype, device=device, requires_grad=True)
    value = torch.randn(kv_shape, dtype=dtype, device=device, requires_grad=True)

    # Clone for separate computation paths
    q_flex = query.clone().detach().requires_grad_(True)
    k_flex = key.clone().detach().requires_grad_(True)
    v_flex = value.clone().detach().requires_grad_(True)

    q_sdpa = query.clone().detach().requires_grad_(True)
    k_sdpa = key.clone().detach().requires_grad_(True)
    v_sdpa = value.clone().detach().requires_grad_(True)

    # Forward pass
    output_flex = attention(
        q_flex, k_flex, v_flex, causal=config.causal, provider="flex"
    )
    output_sdpa = attention(
        q_sdpa, k_sdpa, v_sdpa, causal=config.causal, provider="sdpa"
    )

    # Compare forward outputs
    print(
        f"Forward pass - Max absolute diff: {adiff(output_flex, output_sdpa):.2e}, Mean relative diff: {rdiff(output_flex, output_sdpa):.2e}"
    )

    flex_loss = output_flex.sum()
    sdpa_loss = output_sdpa.sum()

    flex_loss.backward()
    sdpa_loss.backward()

    # Compare gradients
    for name, grad_flex, grad_sdpa in [
        ("query", q_flex.grad, q_sdpa.grad),
        ("key", k_flex.grad, k_sdpa.grad),
        ("value", v_flex.grad, v_sdpa.grad),
    ]:
        print(
            f"{name} grad - Max absolute diff: {adiff(grad_flex, grad_sdpa):.2e}, Mean relative diff: {rdiff(grad_flex, grad_sdpa):.2e}"
        )


if __name__ == "__main__":
    pydra.run(main)
