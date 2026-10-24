"""Convert an adapter's action units/layout into the model RTC tensor contract.

The normalization callback receives only active action channels [L, D_real].
It must return those same channels in model space, without clipping. Relative
action anchors and hardware units belong to the caller, never the RTC sampler.
"""

from collections.abc import Callable

import torch

from wall_x.model.core.action.rtc import RTCCondition
from wall_x.model.core.action.rtc_contract import prepare_condition


def build_condition(
    raw_prefix,
    *,
    normalize_prefix: Callable[[torch.Tensor], torch.Tensor],
    dof_mask,
    horizon,
    action_dim,
    device,
    committed_steps,
    guidance_horizon,
    max_guidance=10.0,
    guidance_mode="lerobot",
):
    """Normalize first, map active channels, then pad with model-space zeros."""
    if dof_mask.ndim != 3 or dof_mask.shape[0] != 1 or dof_mask.shape[-1] != action_dim:
        raise ValueError("RTC preprocessing requires a [1,1 or H,D] DOF mask")
    if dof_mask.shape[1] not in (1, horizon):
        raise ValueError("RTC DOF mask horizon mismatch")
    rows = dof_mask.to(device=device, dtype=torch.float32).reshape(-1, action_dim)
    if not ((rows == 0) | (rows == 1)).all() or not (rows == rows[:1]).all():
        raise ValueError("RTC requires a fixed binary DOF layout over the horizon")
    active = rows[0].bool()
    committed, end = committed_steps, guidance_horizon
    raw = torch.as_tensor(raw_prefix, dtype=torch.float32, device=device)
    if (
        not isinstance(committed, int)
        or isinstance(committed, bool)
        or not isinstance(end, int)
        or isinstance(end, bool)
        or not 1 <= committed < end < horizon
        or raw.ndim != 2
        or raw.shape[0] not in (end, horizon)
        or not torch.isfinite(raw).all()
    ):
        raise ValueError(
            "RTC needs a finite guidance-length or legacy full-horizon prefix"
        )
    if raw.shape[1] != int(active.sum()):
        raise ValueError("RTC prefix and active action dimensions disagree")
    raw = raw[:end]
    normalized = normalize_prefix(raw)
    if not isinstance(normalized, torch.Tensor) or normalized.shape != raw.shape:
        raise ValueError("RTC normalizer must preserve the active prefix shape")
    prefix = torch.zeros((1, horizon, action_dim), device=device)
    prefix[0, : len(raw), active] = normalized.to(device=device, dtype=torch.float32)
    weights = torch.zeros((1, horizon, 1), device=device)
    weights[:, :committed] = 1
    if guidance_mode == "lerobot":
        # Preserve LeRobot's CPU linspace rounding before device transfer.
        weights[0, committed:end, 0] = torch.linspace(1, 0, end - committed + 2)[
            1:-1
        ].to(device)
    else:
        weights[0, committed:end, 0] = torch.arange(
            end - committed, 0, -1, device=device
        ) / (end - committed + 1)
    condition = RTCCondition(prefix, weights, max_guidance, guidance_mode)
    return prepare_condition(condition, prefix)


def make_condition(
    request, normalizer, norm_key, dof_mask, horizon, action_dim, device
):
    """Compatibility adapter for the existing min/delta RTC wire request.

    New normalizer types use build_condition with an explicit conversion.
    Legacy requests without guidance_mode retain their PI VJP interpretation.
    """
    if request is None or request["prefix"] is None:
        return None

    def normalize(raw):
        active = dof_mask.to(device).reshape(-1, action_dim)[0].bool()
        minimum = normalizer.min[norm_key].to(device).reshape(-1)
        delta = normalizer.delta[norm_key].to(device).reshape(-1)
        if minimum.shape != delta.shape:
            raise ValueError("RTC action normalizer dimensions disagree")
        if minimum.numel() == action_dim:
            minimum, delta = minimum[active], delta[active]
        if minimum.numel() != raw.shape[1]:
            raise ValueError("RTC prefix and action normalizer dimensions disagree")
        if (
            not torch.isfinite(minimum).all()
            or not torch.isfinite(delta).all()
            or (delta <= 0).any()
        ):
            raise ValueError("RTC requires finite positive affine normalization scales")
        return 2 * (raw - minimum) / delta - 1

    return build_condition(
        request["prefix"],
        normalize_prefix=normalize,
        dof_mask=dof_mask,
        horizon=horizon,
        action_dim=action_dim,
        device=device,
        committed_steps=request["committed_steps"],
        guidance_horizon=request["guidance_horizon"],
        max_guidance=request["max_guidance"],
        guidance_mode=request.get("guidance_mode", "pi_vjp"),
    )
