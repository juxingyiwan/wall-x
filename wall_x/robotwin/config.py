"""Typed RoboTwin extensions; all shared Wall-X config classes stay untouched."""

import dataclasses
from dataclasses import dataclass

from wall_x.config.data_config import LeRobotDataConfig
from wall_x.config.hyperparams_config import TrainHyperParams
from wall_x.config.infra_config import LoggingConfig
from wall_x.config.loader import load_config as _native_load_config
from wall_x.robotwin.normalization import resolve_normalization


@dataclass
class RobotwinDataConfig(LeRobotDataConfig):
    episode_split_path: str | None = None


@dataclass
class RobotwinLegacyActionDataConfig(RobotwinDataConfig):
    # Persist the historical training-label-only flag without changing its
    # inference-normalizer semantics into the new explicit policy.
    clip_action_targets: bool = True


@dataclass
class RobotwinHyperParams(TrainHyperParams):
    max_training_steps: int | None = None

    def __post_init__(self):
        super().__post_init__()
        if self.max_training_steps is not None and (
            not isinstance(self.max_training_steps, int)
            or isinstance(self.max_training_steps, bool)
            or self.max_training_steps <= 0
        ):
            raise ValueError("max_training_steps must be a positive integer or None")

    @property
    def training_step_limit(self) -> int:
        """Earliest configured update cap, without modifying the LR scheduler."""
        scheduled = int(getattr(self.scheduler, "num_training_steps", 0) or 0)
        caps = [
            n
            for n in (scheduled, self.max_training_steps, self.max_optimizer_steps)
            if n is not None and n > 0
        ]
        return min(caps) if caps else 0


@dataclass
class RobotwinLoggingConfig(LoggingConfig):
    save_on_completed_steps: bool = False


def load_config(config_path, cli_args=None):
    cfg = _native_load_config(config_path, cli_args=cli_args)
    raw = cfg._raw_yaml
    if (cfg.data.lerobot_config or {}).get("adapter") != "robotwin":
        raise ValueError("Use this entry only for data.lerobot_config.adapter=robotwin")
    policy = resolve_normalization(raw)
    raw_data = raw.get("data", {})
    legacy_flag = (
        "clip_action_targets" in raw_data
        and "normalization" not in cfg.data.lerobot_config
    )
    data_cls = RobotwinLegacyActionDataConfig if legacy_flag else RobotwinDataConfig
    extra = (
        {"clip_action_targets": raw_data["clip_action_targets"]} if legacy_flag else {}
    )
    cfg.data = data_cls(
        **dataclasses.asdict(cfg.data),
        episode_split_path=raw_data.get("episode_split_path"),
        **extra,
    )
    # New policies are explicit; legacy action-only checkpoints keep their
    # original flag and therefore their original inference behavior.
    cfg.data.lerobot_config = dict(cfg.data.lerobot_config)
    if not legacy_flag:
        cfg.data.lerobot_config["normalization"] = dataclasses.asdict(policy)
    hp = {
        f.name: getattr(cfg.hyperparams, f.name)
        for f in dataclasses.fields(cfg.hyperparams)
    }
    cfg.hyperparams = RobotwinHyperParams(
        **hp, max_training_steps=raw.get("hyperparams", {}).get("max_training_steps")
    )
    cfg.logging = RobotwinLoggingConfig(
        **dataclasses.asdict(cfg.logging),
        save_on_completed_steps=raw.get("logging", {}).get(
            "save_on_completed_steps", False
        ),
    )
    return cfg
