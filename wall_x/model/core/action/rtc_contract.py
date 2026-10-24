"""Model RTC contract, independent of robot layouts, units and control rates."""

import math
from dataclasses import dataclass, replace

import torch

from .rtc import RTCCondition

NOISE_AT_ZERO = "noise_at_zero_action_at_one"


@dataclass(frozen=True)
class ModelRTCCapability:
    supported: bool = False
    guidance_modes: tuple[str, ...] = ()
    time_convention: str = NOISE_AT_ZERO
    max_batch_size: int = 1
    attention_export: bool = False
    backends: tuple[str, ...] = ()
    input_gradient_modes: tuple[str, ...] = ("pi_vjp",)
    condition_space: str = "normalized_model_action"
    unsupported_reason: str | None = "Model has not declared RTC support"

    def validate(
        self,
        condition,
        *,
        batch_size,
        output_attentions=False,
        backend="native_pytorch",
    ):
        if not self.supported or backend not in self.backends:
            raise ValueError(
                f"RTC unavailable for {backend}: {self.unsupported_reason}"
            )
        if not isinstance(condition, RTCCondition):
            raise TypeError("rtc_condition must be a model-space RTCCondition")
        if condition.guidance_mode not in self.guidance_modes:
            raise ValueError(
                f"Unsupported RTC guidance_mode: {condition.guidance_mode!r}"
            )
        if not 1 <= batch_size <= self.max_batch_size:
            raise ValueError(f"RTC supports batch sizes 1..{self.max_batch_size}")
        if output_attentions and not self.attention_export:
            raise ValueError("RTC does not support attention-map export")
        if (
            condition.guidance_mode in self.input_gradient_modes
            and torch.is_inference_mode_enabled()
        ):
            raise RuntimeError("pi_vjp RTC requires no_grad, not torch.inference_mode")


def prepare_condition(condition, x):
    """Validate once per chunk and transfer caller tensors into sampler space."""
    if not isinstance(condition, RTCCondition):
        raise TypeError("rtc_condition must be a model-space RTCCondition")
    prefix, weights = condition.prefix, condition.weights
    if not isinstance(prefix, torch.Tensor) or not isinstance(weights, torch.Tensor):
        raise TypeError("RTC prefix and weights must be tensors")
    if prefix.shape != x.shape or weights.shape not in (
        (1, x.shape[1], 1),
        (x.shape[0], x.shape[1], 1),
    ):
        raise ValueError("RTC prefix must be [B,H,D] and weights [1 or B,H,1]")
    if not prefix.is_floating_point() or not weights.is_floating_point():
        raise TypeError("RTC prefix and weights must be floating-point tensors")
    prefix = prefix.detach().to(device=x.device, dtype=x.dtype)
    weights = weights.detach().to(device=x.device, dtype=x.dtype)
    if (
        not torch.isfinite(prefix).all()
        or not torch.isfinite(weights).all()
        or (weights < 0).any()
        or (weights > 1).any()
    ):
        raise ValueError("RTC requires finite prefix and weights in [0,1]")
    if not math.isfinite(condition.max_guidance) or condition.max_guidance < 0:
        raise ValueError("RTC max_guidance must be finite and nonnegative")
    if condition.guidance_mode not in ("lerobot", "pi_vjp"):
        raise ValueError(f"Unsupported RTC guidance_mode: {condition.guidance_mode!r}")
    return replace(condition, prefix=prefix, weights=weights)


def validate_rtc_times(times, convention):
    """Adapters must convert reverse-time models before entering this sampler."""
    if convention != NOISE_AT_ZERO:
        raise ValueError("RTC requires noise-at-zero/action-at-one flow time")
    if (
        times.ndim != 1
        or times.numel() < 2
        or not torch.isfinite(times).all()
        or times[0] != 0
        or times[-1] > 1
        or not (times[1:] > times[:-1]).all()
    ):
        raise ValueError("RTC times must strictly increase from 0 to at most 1")
