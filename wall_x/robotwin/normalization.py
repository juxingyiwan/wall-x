"""Versioned action/state normalization owned only by the RoboTwin adapter.

Quantiles set the scale, not the legal range. Old checkpoints without an explicit
policy keep clipping; new checkpoints store both switches under lerobot_config.
"""

from copy import deepcopy
from dataclasses import dataclass

import torch

from wall_x.model.core.action.normalizer import Normalizer


@dataclass(frozen=True)
class NormalizationPolicy:
    clip_action_targets: bool = True
    clip_state_inputs: bool = True

    def __post_init__(self):
        for key in ("clip_action_targets", "clip_state_inputs"):
            if not isinstance(getattr(self, key), bool):
                raise ValueError(f"RoboTwin normalization.{key} must be a boolean")

    @property
    def is_legacy(self):
        return self.clip_action_targets and self.clip_state_inputs


def resolve_normalization(config):
    """Read one policy for training, direct inference and websocket inference."""
    data = config.get("data", {}) or {}
    lerobot = data.get("lerobot_config", {}) or {}
    if lerobot.get("adapter") != "robotwin":
        raise ValueError(
            "RoboTwin normalization requires lerobot_config.adapter=robotwin"
        )
    values = lerobot.get("normalization", {})
    if not isinstance(values, dict):
        raise ValueError("RoboTwin normalization must be a mapping")
    unknown = set(values) - {"clip_action_targets", "clip_state_inputs"}
    if unknown:
        raise ValueError(f"Unknown RoboTwin normalization options: {sorted(unknown)}")
    # Compatibility for the earlier, action-only experimental configuration.
    legacy = data.get("clip_action_targets", True)
    if not isinstance(legacy, bool):
        raise ValueError("data.clip_action_targets must be a boolean")
    if (
        "clip_action_targets" in data
        and "clip_action_targets" in values
        and legacy != values["clip_action_targets"]
    ):
        raise ValueError("Conflicting legacy and RoboTwin action clipping options")
    policy = NormalizationPolicy(
        clip_action_targets=values.get("clip_action_targets", legacy),
        clip_state_inputs=values.get("clip_state_inputs", True),
    )
    if not policy.clip_action_targets and (
        data.get("use_fast_tokenizer", False) or config.get("use_fast_tokenizer", False)
    ):
        raise ValueError("Unclipped actions require use_fast_tokenizer=False")
    task = config.get("task", {}) or {}
    discrete_state = config.get(
        "use_state_string_representation",
        task.get(
            "use_state_string_representation",
            data.get("use_state_string_representation", False),
        ),
    )
    if not policy.clip_state_inputs and discrete_state:
        raise ValueError(
            "Unclipped states require continuous state inputs, not state string representation"
        )
    return policy


def scale_quantiles(values, minimum, delta, *, clip):
    """Match native affine arithmetic and its zero-range fallback exactly."""
    delta = torch.where(delta == 0, torch.ones_like(delta), delta)
    result = (values - minimum) / delta
    result = result * 2 - 1
    return torch.clamp(result, -1, 1) if clip else result


class RobotwinNormalizer(Normalizer):
    """Keep native parameter names and inverse; only forward clipping differs."""

    def __init__(self, source, *, clip):
        torch.nn.Module.__init__(self)
        self.min = deepcopy(source.min)
        self.delta = deepcopy(source.delta)
        self.min_key = source.min_key
        self.delta_key = source.delta_key
        self.clip = clip

    def normalize_data(self, xs, dataset_names):
        names = [name for name in dataset_names if name != "x2_multimodal"]
        if len(names) != len(xs):
            raise ValueError("RoboTwin normalization needs one dataset name per sample")
        return torch.stack(
            [
                scale_quantiles(x, self.min[name], self.delta[name], clip=self.clip)
                for x, name in zip(xs, names, strict=True)
            ]
        )


def adapt_normalizers(config, action, state):
    policy = resolve_normalization(config)
    # The historical top-level flag changed training labels only. Keep its
    # inference padding normalizer unchanged when loading those old configs.
    explicit = "normalization" in config.get("data", {}).get("lerobot_config", {})
    if not policy.clip_action_targets and explicit:
        if action is None:
            raise ValueError(
                "Unclipped RoboTwin actions require saved normalization statistics"
            )
        action = RobotwinNormalizer(action, clip=False)
    if not policy.clip_state_inputs:
        if state is None:
            raise ValueError(
                "Unclipped RoboTwin states require saved normalization statistics"
            )
        state = RobotwinNormalizer(state, clip=False)
    return action, state
