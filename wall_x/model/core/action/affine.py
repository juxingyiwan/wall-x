"""Explicit FP32 affine scaling, with clipping independent of the scale."""

import torch


def scale(value, lower, span, *, clip):
    value = value.float()
    lower = torch.as_tensor(lower, dtype=torch.float32, device=value.device)
    span = torch.as_tensor(span, dtype=torch.float32, device=value.device)
    result = 2 * (value - lower) / span - 1
    return result.clamp(-1, 1) if clip else result


def unscale(value, lower, span):
    value = value.float()
    return (value + 1) * torch.as_tensor(
        span, dtype=torch.float32, device=value.device
    ) / 2 + torch.as_tensor(lower, dtype=torch.float32, device=value.device)
