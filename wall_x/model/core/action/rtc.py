"""Request-scoped RTC guidance for rectified flow (noise at t=0, action at t=1).

``lerobot`` matches the effective direct-residual update in LeRobot b9cb121.
``pi_vjp`` retains Physical Intelligence's full endpoint-Jacobian update.
Select the guidance mode explicitly; the two updates are not interchangeable.
"""

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class RTCCondition:
    prefix: torch.Tensor  # normalized [B, H, D], including virtual channels
    weights: torch.Tensor  # [1, H, 1], committed prefix then linear taper
    max_guidance: float = 10.0
    guidance_mode: str = "lerobot"


def guided_velocity(denoise, x, t, condition, dof_mask):
    """Apply the explicitly selected endpoint guidance, without parameter grads.

    LeRobot b9cb121 enables input gradients *after* calling its denoiser, so its
    effective endpoint derivative is identity. Compute that residual directly
    under no_grad, avoiding an unused forward graph and a denoiser backward.
    Wall-X time runs noise -> action, opposite to LeRobot; hence the plus sign.
    Both modes are soft guidance, not a hard overwrite of predicted actions.
    """
    if condition.guidance_mode == "lerobot":
        with torch.no_grad():
            x = x.detach()
            velocity = denoise(t, x)
            endpoint = x + (1 - t) * velocity
            correction = (condition.prefix - endpoint) * condition.weights * dof_mask
    elif condition.guidance_mode == "pi_vjp":
        if torch.is_inference_mode_enabled():
            raise RuntimeError("pi_vjp RTC requires no_grad, not torch.inference_mode")
        with torch.enable_grad():
            x = x.detach().requires_grad_(True)
            velocity = denoise(t, x)
            endpoint = x + (1 - t) * velocity
            error = (
                (condition.prefix - endpoint) * condition.weights * dof_mask
            ).detach()
            correction = torch.autograd.grad(endpoint, x, grad_outputs=error)[0]
    else:
        raise ValueError(f"Unsupported RTC guidance_mode: {condition.guidance_mode!r}")
    with torch.no_grad():
        if condition.guidance_mode == "lerobot":
            # Match the reference operation order as well as its algebra.
            # Reassociating this ratio, or eliding the time reversal roundtrip,
            # can change BF16 denoising decisions over successive flow steps.
            tau = 1 - (1 - t)
            squared = (1 - tau) ** 2
            inv_r2 = (squared + tau**2) / squared
            c = torch.nan_to_num((1 - tau) / tau, posinf=condition.max_guidance)
            gain = torch.nan_to_num(c * inv_r2, posinf=condition.max_guidance)
            gain = gain.clamp(max=condition.max_guidance)
        else:
            gain = (t.square() + (1 - t).square()) / (t * (1 - t)).clamp_min(1e-8)
            gain = gain.clamp(max=condition.max_guidance)
        result = velocity + gain * correction * dof_mask
    if not torch.isfinite(result).all():
        raise RuntimeError("RTC produced a non-finite velocity")
    return result.detach()


def make_condition(
    request, normalizer, norm_key, dof_mask, horizon, action_dim, device
):
    """Compatibility import; action preprocessing lives outside the RTC core."""
    from wall_x.model.adapters.rtc_preprocessing import make_condition as preprocess

    return preprocess(
        request, normalizer, norm_key, dof_mask, horizon, action_dim, device
    )
