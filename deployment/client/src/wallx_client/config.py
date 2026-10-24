"""Strict, side-effect-free configuration loading."""

import math
import re
import warnings
from dataclasses import asdict, dataclass, field, fields, replace
from pathlib import Path
from urllib.parse import urlparse

import yaml


@dataclass(frozen=True)
class PolicyConfig:
    uri: str = "ws://127.0.0.1:32195"
    checkpoint_id: str = ""
    checkpoint_sha256: str = ""
    horizon: int = 32
    action_dims: tuple[int, ...] = (6, 1, 19)
    camera_names: tuple[str, ...] = ("front_view", "side_view")
    connect_timeout_s: float = 10.0
    request_timeout_s: float = 15.0
    max_message_bytes: int = 8 * 1024 * 1024
    jpeg_quality: int = 95
    padding: str = "zero"  # Expected model policy for either deployment backend.


@dataclass(frozen=True)
class RobotConfig:
    can_channel: str = "can0"
    robot_id: str = "follower1"
    camera_devices: dict[str, str] = field(default_factory=dict)
    # One buffer can starve UVC capture during MJPEG decoding.
    camera_buffer_size: int = 2
    camera_mapping_verified: bool = False
    width: int = 640
    height: int = 480
    fps: int = 30
    max_camera_age_s: float = 0.15
    max_camera_skew_s: float = 0.05
    camera_recovery_timeout_s: float = 1.0
    camera_max_consecutive_failures: int = 3
    camera_max_reopens: int = 3
    max_feedback_age_s: float = 0.12
    min_temperature_c: float = -25.0
    max_temperature_c: float = 65.0
    temperature_feedback_grace_s: float = 0.5
    lock_paths: tuple[str, ...] = ()


@dataclass(frozen=True)
class ControlConfig:
    # When omitted, rtc.enabled selects sync or rtc_fifo.
    executor: str | None = None
    hz: int = 30
    execution_horizon: int = 16
    duration_s: float = 300.0
    max_policy_chunks: int = 100000
    max_tick_s: float = 0.15
    max_response_age_s: float = 2.0
    max_state_drift_arm_deg: float = 2.0
    max_state_drift_gripper_deg: float = 15.0
    # Arrival tolerance for startup/shutdown gripper homing (physical degrees).
    settle_gripper_deg: float = 3.0
    startup_home: bool = True
    startup_gripper_speed_deg_s: float = 60.0


@dataclass(frozen=True)
class ShutdownConfig:
    worker_timeout_s: float = 2.0
    # Native arm path followed by verified gripper zero, using existing calibration.
    home_timeout_s: float = 65.0
    arm_tolerance_deg: float = 2.0
    settle_s: float = 0.25
    # A stale-only fault may recover for homing, never for resuming the episode.
    feedback_recovery_timeout_s: float = 1.0


@dataclass(frozen=True)
class RTCConfig:
    # RTC is opt-in; otherwise chunks execute synchronously.
    enabled: bool = False
    model_dt_s: float = 1 / 30
    latency_budget_s: float = 0.35
    guard_s: float = 0.1
    blend_steps: int = 6
    replan_interval_s: float = 0.35
    max_guidance: float = 10.0
    guidance_mode: str = "lerobot"


@dataclass(frozen=True)
class Config:
    task: str
    output_root: str
    policy: PolicyConfig
    robot: RobotConfig
    control: ControlConfig
    display_data: bool = True
    save_rollout: bool = True
    display_ip: str | None = None
    display_port: int = 9876
    rtc: RTCConfig = field(default_factory=RTCConfig)
    shutdown: ShutdownConfig = field(default_factory=ShutdownConfig)

    def to_dict(self) -> dict:
        return asdict(self)

    @property
    def execution_mode(self) -> str:
        if self.control.executor is not None:
            return self.control.executor
        if not self.rtc.enabled:
            return "sync"
        return "rtc_fifo"


def resolve_execution(cfg: Config) -> Config:
    """Explicit executor owns activation; otherwise use rtc.enabled."""
    return replace(cfg, rtc=replace(cfg.rtc, enabled=cfg.execution_mode != "sync"))


def check_legacy_continuous(value, source):
    """Accept only the retired switch's inactive value, without selecting a mode."""
    if type(value) is not bool:
        raise ValueError(f"{source} must be a boolean")
    if value:
        raise ValueError(
            f"{source}=true has been removed. Remove this option and explicitly choose "
            "rtc.enabled=false (--rtc false) for synchronous execution or "
            "rtc.enabled=true (--rtc true) for guided RTC. No automatic conversion is made."
        )
    warnings.warn(
        f"{source}=false is deprecated and ignored; remove this option. "
        "Only rtc.enabled (--rtc) selects the execution mode.",
        FutureWarning,
        stacklevel=2,
    )


def _section(cls, value):
    if not isinstance(value, dict):
        raise ValueError(f"{cls.__name__} must be a mapping")
    if cls is ControlConfig and "continuous" in value:
        value = dict(value)
        check_legacy_continuous(value.pop("continuous"), "control.continuous")
    names = {f.name for f in fields(cls)}
    if unknown := set(value) - names:
        raise ValueError(f"Unknown {cls.__name__} keys: {sorted(unknown)}")
    value = dict(value)
    for name in ("action_dims", "camera_names", "lock_paths"):
        if name in value:
            value[name] = tuple(value[name])
    return cls(**value)


def load_config(path: str | Path, *, validate=True) -> Config:
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    required = {
        "task",
        "output_root",
        "policy",
        "robot",
        "control",
    }
    optional = {"display_data", "save_rollout", "display_ip", "display_port", "rtc", "shutdown"}
    if not isinstance(raw, dict) or not required <= set(raw) or set(raw) - required - optional:
        raise ValueError(
            "Config requires task, output_root, policy, robot, control; unknown keys rejected"
        )
    cfg = Config(
        raw["task"],
        raw["output_root"],
        _section(PolicyConfig, raw["policy"]),
        _section(RobotConfig, raw["robot"]),
        _section(ControlConfig, raw["control"]),
        rtc=_section(RTCConfig, raw.get("rtc", {})),
        shutdown=_section(ShutdownConfig, raw.get("shutdown", {})),
        **{name: raw[name] for name in optional - {"rtc", "shutdown"} if name in raw},
    )
    if validate:
        cfg = resolve_execution(cfg)
        validate_config(cfg)
    return cfg


def validate_config(cfg: Config) -> None:
    for name in ("display_data", "save_rollout"):
        if type(getattr(cfg, name)) is not bool:
            raise ValueError(f"{name} must be a YAML boolean")
    if cfg.display_ip is not None and (
        not isinstance(cfg.display_ip, str) or not cfg.display_ip.strip()
    ):
        raise ValueError("display_ip must be a nonempty host name")
    if type(cfg.display_port) is not int or not 1 <= cfg.display_port <= 65535:
        raise ValueError("display_port must be in 1..65535")
    if not isinstance(cfg.task, str) or not cfg.task.strip():
        raise ValueError("task must contain the full training instruction")
    if not isinstance(cfg.output_root, str) or not cfg.output_root:
        raise ValueError("output_root is required")
    p, r, c = cfg.policy, cfg.robot, cfg.control
    if not isinstance(p.padding, str) or p.padding not in ("zero", "original"):
        raise ValueError("policy.padding must be zero or original")
    if c.executor not in (None, "sync", "rtc_fifo"):
        raise ValueError("control.executor must be sync or rtc_fifo")
    for name in ("max_policy_chunks",):
        if type(getattr(c, name)) is not int or getattr(c, name) < 1:
            raise ValueError(f"control.{name} must be a positive integer")
    for name, value in asdict(cfg.shutdown).items():
        if type(value) not in (int, float) or not math.isfinite(value) or value <= 0:
            raise ValueError(f"shutdown.{name} must be finite and positive")
    if cfg.shutdown.settle_s >= cfg.shutdown.home_timeout_s:
        raise ValueError("shutdown.settle_s must be shorter than home_timeout_s")
    if not cfg.shutdown.settle_s < cfg.shutdown.feedback_recovery_timeout_s <= 5:
        raise ValueError(
            "shutdown.feedback_recovery_timeout_s must exceed settle_s and be at most 5s"
        )
    rtc = cfg.rtc
    if type(rtc.enabled) is not bool:
        raise ValueError("rtc.enabled must be a boolean")
    if rtc.guidance_mode not in ("lerobot", "pi_vjp"):
        raise ValueError("rtc.guidance_mode must be lerobot or pi_vjp")
    for name in (
        "model_dt_s",
        "latency_budget_s",
        "guard_s",
        "replan_interval_s",
        "max_guidance",
    ):
        value = getattr(rtc, name)
        if type(value) not in (int, float) or not math.isfinite(value) or value <= 0:
            raise ValueError(f"rtc.{name} must be finite and positive")
    for name in ("blend_steps",):
        if type(getattr(rtc, name)) is not int:
            raise ValueError(f"rtc.{name} must be an integer")
    if cfg.execution_mode != "sync":
        if not 1 <= rtc.blend_steps <= 16:
            raise ValueError("rtc.blend_steps must be in 1..16")
        if not math.isclose(rtc.model_dt_s, 1 / 30, abs_tol=1e-8) or rtc.max_guidance > 20:
            raise ValueError("RTC v1 requires a 30 Hz model grid and max_guidance <= 20")
    uri = urlparse(p.uri)
    if p.uri != "local://wall-x" and (uri.scheme not in {"ws", "wss"} or not uri.hostname):
        raise ValueError("policy.uri must be a WebSocket URI or local://wall-x")
    if p.checkpoint_sha256 and not re.fullmatch(r"[0-9a-f]{64}", p.checkpoint_sha256):
        raise ValueError("checkpoint_sha256 must be a lowercase SHA256")
    if p.horizon != 32 or p.action_dims != (6, 1, 19):
        raise ValueError("This adapter requires the 32 × (6+1+19) action layout")
    if p.camera_names != ("front_view", "side_view"):
        raise ValueError("v0.1 expects front_view and side_view in training order")
    if set(r.camera_devices) != set(p.camera_names):
        raise ValueError("Each training camera requires one explicit device mapping")
    if type(r.camera_buffer_size) is not int or r.camera_buffer_size not in (1, 2):
        raise ValueError("camera_buffer_size must be 1 or 2")
    if type(r.camera_max_consecutive_failures) is not int or r.camera_max_consecutive_failures < 1:
        raise ValueError("camera_max_consecutive_failures must be a positive integer")
    if type(r.camera_max_reopens) is not int or not 0 <= r.camera_max_reopens <= 10:
        raise ValueError("camera_max_reopens must be an integer in 0..10")
    if len(set(r.camera_devices.values())) != len(r.camera_devices):
        raise ValueError("Camera devices must be distinct")
    grace = r.temperature_feedback_grace_s
    if type(grace) not in (int, float) or not math.isfinite(grace) or not 0 <= grace <= 1:
        raise ValueError("robot.temperature_feedback_grace_s must be finite in 0..1 (0 disables)")
    minimum = r.min_temperature_c
    if type(minimum) not in (int, float) or not math.isfinite(minimum):
        raise ValueError("robot.min_temperature_c must be finite")
    if type(r.camera_mapping_verified) is not bool or type(c.startup_home) is not bool:
        raise ValueError("camera_mapping_verified and startup_home must be YAML booleans")
    if type(c.execution_horizon) is not int or (
        cfg.execution_mode == "sync" and not 1 <= c.execution_horizon <= p.horizon
    ):
        raise ValueError("execution_horizon must be an integer in 1..32")
    if c.hz != 30 or r.fps != 30 or (r.width, r.height) != (640, 480):
        raise ValueError("This B601 baseline uses 30 Hz control and 640×480 at 30 FPS")
    if not r.lock_paths or len(set(r.lock_paths)) != len(r.lock_paths):
        raise ValueError("Configure distinct shared robot lock paths")
    if not 1 <= p.jpeg_quality <= 100:
        raise ValueError("jpeg_quality must be in 1..100")
    for name, value in {
        **{f"policy.{k}": v for k, v in asdict(p).items() if k.endswith("_s")},
        **{f"robot.{k}": v for k, v in asdict(r).items() if k.endswith(("_s", "_c"))},
        **{f"control.{k}": v for k, v in asdict(c).items() if k.endswith(("_s", "_deg", "_s2"))},
    }.items():
        if name in {"robot.temperature_feedback_grace_s", "robot.min_temperature_c"}:
            continue
        if type(value) not in (float, int) or not math.isfinite(value) or value <= 0:
            raise ValueError(f"{name} must be finite and positive")
    if minimum >= r.max_temperature_c - 5:
        raise ValueError("robot.min_temperature_c must be below max_temperature_c minus 5")
    if p.max_message_bytes < 1024:
        raise ValueError("max_message_bytes is too small")


def require_execution_ready(cfg: Config) -> None:
    if not cfg.robot.camera_mapping_verified:
        raise ValueError(
            "Verify top/wrist against dataset front/side, then set camera_mapping_verified"
        )
    if not cfg.policy.checkpoint_sha256 or cfg.policy.checkpoint_sha256 == "0" * 64:
        raise ValueError("Pin the checkpoint SHA256 before enabling motors")
