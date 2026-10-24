"""Versioned, stateless RTC wire contract. Enabled only by a client request."""

import math

import numpy as np


def validate_request(value, capability, mode):
    if value is None:
        return None
    if not isinstance(value, dict) or type(value.get("enabled")) is not bool:
        raise ValueError("rtc.enabled must be a boolean")
    if not value["enabled"]:
        return None
    if not capability.get("supported") or mode != "flow":
        raise ValueError(
            "RTC unavailable: requires X2ROBOT, raw flow actions, single-request vanilla backend without server smoothing"
        )
    expected = {
        "version",
        "enabled",
        "epoch",
        "plan_id",
        "model_dt_s",
        "prefix",
        "committed_steps",
        "guidance_horizon",
        "max_guidance",
    }
    if (
        set(value) not in (expected, expected | {"guidance_mode"})
        or value["version"] != 1
    ):
        raise ValueError("Invalid RTC v1 fields")
    if value.get("guidance_mode", "pi_vjp") not in ("lerobot", "pi_vjp"):
        raise ValueError("RTC guidance_mode must be lerobot or pi_vjp")
    for key in ("epoch", "plan_id", "committed_steps", "guidance_horizon"):
        if type(value[key]) is not int or value[key] < 0:
            raise ValueError(f"rtc.{key} must be a nonnegative integer")
    if type(value["model_dt_s"]) not in (int, float) or not math.isclose(
        value["model_dt_s"], capability["model_dt_s"], abs_tol=1e-8
    ):
        raise ValueError("RTC model timebase mismatch")
    weight = value["max_guidance"]
    if (
        type(weight) not in (int, float)
        or not math.isfinite(weight)
        or not 0 < weight <= 20
    ):
        raise ValueError("RTC max_guidance must be in (0,20]")
    if value["prefix"] is None:
        if value["committed_steps"] or value["guidance_horizon"]:
            raise ValueError("RTC bootstrap cannot commit a prefix")
    else:
        prefix = np.asarray(value["prefix"], dtype=np.float32)
        horizon = capability["action_horizon"]
        if (
            prefix.shape
            not in (
                (value["guidance_horizon"], capability["real_action_dim"]),
                (horizon, capability["real_action_dim"]),  # Legacy full-horizon v1.
            )
            or not np.isfinite(prefix).all()
        ):
            raise ValueError("RTC prefix shape/values invalid")
        if not 1 <= value["committed_steps"] < value["guidance_horizon"] < horizon:
            raise ValueError("RTC needs committed, tapered and free actions")
    return value


def acknowledgement(request):
    result = {
        key: request[key]
        for key in (
            "version",
            "epoch",
            "plan_id",
            "committed_steps",
            "guidance_horizon",
        )
    } | {"mode": "bootstrap" if request["prefix"] is None else "guided"}
    if "guidance_mode" in request:
        result["guidance_mode"] = request["guidance_mode"]
    return result
def build_capability(policy, *, enabled):
    """Build the B601 wire contract from validated checkpoint metadata.

    Pre-manifest absolute B601 checkpoints retain the explicit legacy 30 Hz
    profile. A versioned manifest must supply its validated timebase; it cannot
    inherit that fallback. This executor does not resample other frequencies.
    """
    import math

    metadata = policy.metadata.get("wallx_client", {})
    versioned = "model_action_representation" in metadata
    period = metadata.get("model_dt_s", None if versioned else 1 / 30)
    timebase_ok = type(period) in (int, float) and math.isfinite(period) and math.isclose(period, 1 / 30, rel_tol=0, abs_tol=1e-8)
    supported = bool(enabled and timebase_ok and policy.config.action_horizon == 32
                     and policy.config.action_dim == 26 and metadata.get("state_dim") == 7)
    return {
        "version": 1, "supported": supported, "model_dt_s": period,
        "model_timebase_source": metadata.get("model_timebase_source", "missing" if versioned else "legacy_b601_30hz"),
        "unsupported_reason": None if supported else "RTC requires compatible B601 layout, 30 Hz checkpoint and inference backend",
        "action_horizon": policy.config.action_horizon, "real_action_dim": 7,
        "guidance": "endpoint_vjp", "prefix_units": "recorded_commands",
        "guidance_modes": ["lerobot", "pi_vjp"],
    }
