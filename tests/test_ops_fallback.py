"""Public fallback tests for operator proxies."""

import pytest
import torch

from wall_x.model.core.ops import rmsnorm


def test_rmsnorm_pytorch_fallback_matches_reference():
    x = torch.randn(2, 8, dtype=torch.float32)
    weight = torch.randn(8, dtype=torch.float32)
    eps = 1e-6

    out = rmsnorm.call_with_backend("pytorch", x, weight, eps)

    ref = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps)
    ref = ref * weight
    torch.testing.assert_close(out, ref)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available")
def test_rmsnorm_cuda_inline_matches_pytorch_when_available():
    if "cuda_inline" not in rmsnorm.available_backends():
        pytest.skip("CUDA inline backend is not available")

    x = torch.randn(2, 16, device="cuda", dtype=torch.float16)
    weight = torch.randn(16, device="cuda", dtype=torch.float16)
    eps = 1e-6

    out = rmsnorm.call_with_backend("cuda_inline", x, weight, eps)
    ref = rmsnorm.call_with_backend("pytorch", x, weight, eps)
    torch.testing.assert_close(out, ref, rtol=1e-3, atol=1e-3)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available")
@pytest.mark.parametrize("interleave", [False, True])
def test_installed_rope_cuda_forward_and_backward_match_pytorch(interleave):
    from wall_x.model.core.ops import rope

    torch.manual_seed(17)
    q = torch.randn(2, 4, 3, 32, device="cuda", dtype=torch.float32)
    k = torch.randn(2, 4, 2, 32, device="cuda", dtype=torch.float32)
    angles = torch.randn(2, 4, 16, device="cuda")
    cos, sin = angles.cos(), angles.sin()
    gradients = [torch.randn_like(q), torch.randn_like(k)]
    reference = [q.clone().requires_grad_(), k.clone().requires_grad_()]
    expected = rope.call_with_backend(
        "pytorch", *reference, cos, sin, interleave=interleave
    )
    expected_grads = torch.autograd.grad(expected, reference, gradients)
    inputs = [q.clone().requires_grad_(), k.clone().requires_grad_()]
    actual = rope.call_with_backend(
        "cuda_inline", *inputs, cos, sin, interleave=interleave
    )
    for result, target in zip(actual, expected):
        torch.testing.assert_close(result, target, rtol=1e-5, atol=1e-5)
    actual_grads = torch.autograd.grad(
        actual, inputs, [gradient.clone() for gradient in gradients]
    )
    for result, target in zip(actual_grads, expected_grads):
        torch.testing.assert_close(result, target, rtol=1e-5, atol=1e-5)
