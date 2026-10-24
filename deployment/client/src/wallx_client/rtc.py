"""Client RTC request contract and measured latency budget (no hardware imports)."""

import math
from collections import deque

import numpy as np

from .contracts import ProtocolError


def validate_capability(cap, policy, rtc):
    if not isinstance(cap, dict) or cap.get("supported") is not True:
        raise ProtocolError(
            "Server does not support guided RTC; restart the updated server or select --rtc false"
        )
    if (
        cap.get("version") != 1
        or cap.get("action_horizon") != policy.horizon
        or cap.get("real_action_dim") != 7
        or cap.get("prefix_units") != "recorded_commands"
        or not math.isclose(cap.get("model_dt_s", 0), rtc.model_dt_s, abs_tol=1e-8)
    ):
        raise ProtocolError("RTC capability/timebase mismatch")
    modes = cap.get("guidance_modes")
    if not isinstance(modes, list) or rtc.guidance_mode not in modes:
        raise ProtocolError(
            f"Server does not advertise RTC {rtc.guidance_mode}; restart the updated server"
        )


def normalize_prev_actions_length(prefix, target_steps):
    """Match LeRobot's RTC input: truncate or hold-pad only the guided region."""
    prefix = np.asarray(prefix)
    if prefix.ndim != 2 or not len(prefix):
        raise ValueError("RTC needs a nonempty 2D prefix to hold its last action")
    if target_steps <= 0:
        raise ValueError("RTC guidance length must be positive")
    if len(prefix) >= target_steps:
        return prefix[:target_steps].copy()
    hold = np.repeat(prefix[-1:], target_steps - len(prefix), axis=0)
    return np.concatenate([prefix, hold], axis=0)


def request(cfg, epoch, plan_id, prefix=None, committed=0, *, guidance_horizon=None):
    # Keep a free suffix even when the measured delay grows. The overlap is a
    # preference, not permission to invent model actions beyond the horizon.
    end = 0
    if prefix is not None:
        end = (
            min(committed + cfg.blend_steps, len(prefix) - 1)
            if guidance_horizon is None
            else guidance_horizon
        )
        prefix = normalize_prev_actions_length(prefix, end)
    return dict(
        version=1,
        enabled=True,
        epoch=epoch,
        plan_id=plan_id,
        model_dt_s=cfg.model_dt_s,
        prefix=None if prefix is None else np.asarray(prefix).tolist(),
        committed_steps=committed,
        guidance_horizon=end,
        max_guidance=cfg.max_guidance,
        guidance_mode=cfg.guidance_mode,
    )


def validate_ack(ack, sent):
    expected = {
        key: sent[key]
        for key in ("version", "epoch", "plan_id", "committed_steps", "guidance_horizon")
    }
    expected["mode"] = "bootstrap" if sent["prefix"] is None else "guided"
    if "guidance_mode" in sent:
        expected["guidance_mode"] = sent["guidance_mode"]
    if ack != expected:
        raise ProtocolError("RTC acknowledgement mismatch; response must not be executed")


class LatencyBudget:
    def __init__(self, cfg):
        self.cfg = cfg
        self.samples = deque(maxlen=32)

    def observe(self, seconds):
        if not math.isfinite(seconds) or seconds <= 0:
            raise ValueError("Invalid inference latency")
        self.samples.append(seconds)

    def estimate_seconds(self):
        return max(
            self.cfg.latency_budget_s, float(np.percentile(self.samples, 95)) if self.samples else 0
        )
