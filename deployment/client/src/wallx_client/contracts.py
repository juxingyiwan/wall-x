"""Explicit units and immutable snapshots at the policy/hardware boundary."""

import time
from dataclasses import dataclass

import numpy as np

JOINTS = (
    "shoulder_pan",
    "shoulder_lift",
    "elbow_flex",
    "wrist_flex",
    "wrist_yaw",
    "wrist_roll",
    "gripper",
)
STATE_UNITS = "physical_motor_degrees"
ACTION_UNITS = "recorded_commands_before_driver_sign_and_gripper_scale"


class ClientError(RuntimeError):
    """An actionable client error."""


class ProtocolError(ClientError):
    """The peer or payload violates the configured policy contract."""


class TransportError(ClientError):
    """A connection failed; its pending action must not be executed."""


class HardwareError(ClientError):
    """Feedback, capture, or control failed."""


class CameraUnavailable(HardwareError):
    """No fresh synchronized pair yet; motor feedback is checked independently.

    An optional absolute monotonic deadline is owned by the capture provider,
    which must enforce it before publishing any recovered pair.
    """

    def __init__(self, message, *, recovery_deadline=None):
        super().__init__(message)
        self.recovery_deadline = recovery_deadline


@dataclass(frozen=True)
class Observation:
    state: np.ndarray
    views: dict[str, np.ndarray]
    captured_monotonic: float
    captured_wall: float
    state_monotonic: float
    camera_skew_s: float = 0.0
    capture_sequence: int = 0
    camera_generation: int = 0

    def validate_payload(self, camera_names: tuple[str, ...]) -> None:
        """Check model inputs without imposing a controller-specific age limit."""
        if (
            not isinstance(self.state, np.ndarray)
            or self.state.shape != (7,)
            or self.state.dtype.kind not in "iuf"
            or not np.isfinite(self.state).all()
        ):
            raise ProtocolError("Expected seven finite physical motor angles in degrees")
        if not isinstance(self.views, dict) or set(self.views) != set(camera_names):
            raise ProtocolError(f"Camera keys must be {camera_names}")
        for name, rgb in self.views.items():
            if (
                not isinstance(rgb, np.ndarray)
                or rgb.dtype != np.uint8
                or rgb.ndim != 3
                or rgb.shape[2] != 3
                or rgb.size == 0
            ):
                raise ProtocolError(f"{name}: expected uint8 H×W×3 RGB")

    def validate(self, camera_names: tuple[str, ...], max_age_s: float) -> None:
        self.validate_payload(camera_names)
        now = time.monotonic()
        for label, timestamp in (
            ("image", self.captured_monotonic),
            ("state", self.state_monotonic),
        ):
            if not 0 <= now - timestamp <= max_age_s:
                raise ProtocolError(f"{label} snapshot is stale or has an invalid timestamp")


@dataclass(frozen=True)
class InferenceTiming:
    """Client monotonic times; never subtract timestamps from another host.

    Preparation includes JPEG and backend serialization. The call interval
    includes backend execution, decoding and validation, not recording.
    Server-reported timing is separate and is not a control clock.
    """

    preparation_started_monotonic: float
    dispatched_monotonic: float
    completed_monotonic: float

    @property
    def preparation_s(self) -> float:
        return self.dispatched_monotonic - self.preparation_started_monotonic

    @property
    def call_s(self) -> float:
        return self.completed_monotonic - self.dispatched_monotonic


@dataclass(frozen=True)
class Prediction:
    request_id: str
    generation: int
    actions: np.ndarray
    observation: Observation
    received_monotonic: float
    roundtrip_s: float
    server_timing: dict
    rtc: dict | None = None
    timing: InferenceTiming | None = None

    @property
    def observation_to_result_s(self) -> float:
        return self.received_monotonic - self.observation.captured_monotonic
