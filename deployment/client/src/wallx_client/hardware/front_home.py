"""Plan a continuous inward arm return before the audited native home.

The plane is base_link +X = 0.50 m for the existing gripper_end FK point.
Keep 30 mm of planning clearance; this is not a calibrated collision model.
No CAN, motor-zero writes, or policy-action changes live in the planner.
"""

import logging
import math
import time
from dataclasses import dataclass
from itertools import product

import numpy as np

from ..contracts import HardwareError

FORWARD_LIMIT_M = 0.50
CLEARANCE_M = 0.03
MAX_SPEED_DEG_S = 40.0
MAX_ACCEL_DEG_S2 = 40.0
RATE_SLEW_S = 0.5
PERIOD_S = 1 / 30


def tool_positions(codec, motor_deg):
    return codec.fk(np.deg2rad(motor_deg) + codec.offset)[..., :3, 3]


def _sample_line(a, b):
    count = max(2, math.ceil(float(np.max(np.abs(b - a))) / 0.25) + 1)
    return np.linspace(a, b, count)


def native_path(start):
    middle = start.copy()
    middle[[0, 3, 4, 5]] = 0
    return np.concatenate((_sample_line(start, middle), _sample_line(middle, np.zeros(6))[1:]))


def _joint_reaches(codec):
    """Global FK Lipschitz radii, from downstream URDF translation lengths."""
    radii = np.zeros(6)
    downstream = 0.0
    for origin, index, _axis in reversed(codec.fk.chain):
        if index is not None:
            radii[index] = downstream
        downstream += float(np.linalg.norm(origin[:3, 3]))
    return radii


@dataclass(frozen=True)
class HomePath:
    start: np.ndarray
    bend: np.ndarray
    native: bool
    duration_s: float
    native_max_x_m: float
    planned_max_x_m: float
    start_x_m: float

    def joints(self, progress):
        # Each joint moves monotonically between its start and calibrated zero.
        s = np.asarray(progress)[..., None]
        return self.start * (1 - s - self.bend * s * (1 - s))

    def summary(self):
        return dict(
            mode="native" if self.native else "coordinated_then_native",
            frame="base_link",
            axis="+X",
            point="gripper_end",
            limit_m=FORWARD_LIMIT_M,
            clearance_m=CLEARANCE_M,
            start_x_m=self.start_x_m,
            native_max_x_m=self.native_max_x_m,
            planned_max_x_m=self.planned_max_x_m,
            starting_outside_plane=self.start_x_m > FORWARD_LIMIT_M,
            duration_s=self.duration_s,
            bend=self.bend.tolist(),
        )


def plan_home(codec, measured, limits):
    """Choose the whole route before dispatch; never clip Cartesian commands.

    Starts already in the clearance band or outside the plane must move inward
    without increasing their nominal excursion until they reach the inner band.
    Replanning failures never fall back to a known outward native trajectory.
    """
    measured, limits = np.asarray(measured, float), np.asarray(limits, float)
    if measured.shape != (7,) or limits.shape != (7, 2):
        raise HardwareError("Forward home requires seven motor positions and limits")
    if not np.isfinite(measured).all() or not np.isfinite(limits).all():
        raise HardwareError("Non-finite forward home input")
    if (
        np.any(limits[:, 0] >= limits[:, 1])
        or np.any(limits[:6, 0] > 0)
        or np.any(limits[:6, 1] < 0)
    ):
        raise HardwareError("Calibrated arm zero is outside driver limits")
    if (
        codec.config.get("base_link") != "base_link"
        or codec.config.get("tool_link") != "gripper_end"
    ):
        raise HardwareError("Forward home requires the audited base_link/gripper_end frames")
    start = np.clip(measured[:6], *limits[:6].T)
    if np.max(np.abs(start - measured[:6])) > 2.0:
        raise HardwareError("Forward home start is more than 2deg outside driver limits")
    radii = _joint_reaches(codec)
    initial_xyz = tool_positions(codec, measured[:6])
    zero_xyz = tool_positions(codec, np.zeros(6))
    inner = FORWARD_LIMIT_M - CLEARANCE_M
    if zero_xyz[0] >= inner:
        raise HardwareError("Calibrated arm zero has no forward-plane clearance")

    def inspect(points, *, keep_height=False):
        points = np.vstack((measured[:6], points))
        xyz = tool_positions(codec, points)
        if not np.isfinite(xyz).all():
            raise HardwareError("Non-finite homing forward kinematics")
        # On these componentwise monotone intervals every interior point lies
        # within half the weighted L1 joint span of at least one endpoint.
        # This bounds FK excursion between samples, not only sampled positions.
        padding = np.deg2rad(np.abs(np.diff(points, axis=0))) @ radii / 2
        upper = np.maximum(xyz[:-1, 0], xyz[1:, 0]) + padding
        peak = float(np.max(upper))
        if initial_xyz[0] <= inner:
            # The nominal route keeps the clearance; its conservative numerical
            # bound must fit the hard plane. Do not reject a safe inward start
            # on the clearance line merely because a bound straddles that line.
            valid = peak <= FORWARD_LIMIT_M and bool(np.all(xyz[:, 0] <= inner + 1e-6))
        else:
            # A pre-existing excursion is recovered continuously, not aborted.
            # While outside the inner band require a non-increasing sampled X;
            # once inside, retain at least the inter-sample clearance to 0.50 m.
            ceiling = np.maximum(inner, np.minimum.accumulate(xyz[:, 0])[:-1])
            valid = bool(np.all(xyz[1:, 0] <= ceiling + 1e-6))
            inside = np.maximum(xyz[:-1, 0], xyz[1:, 0]) <= inner
            valid = valid and bool(np.all(upper[inside] <= FORWARD_LIMIT_M))
        if keep_height:
            # Do not introduce a new downward sweep below the lower endpoint.
            floor = min(float(initial_xyz[2]), float(zero_xyz[2])) - 0.005
            lower = np.minimum(xyz[:-1, 2], xyz[1:, 2]) - padding
            valid = valid and bool(np.all(lower >= floor))
        return valid, peak

    original = np.clip(native_path(measured[:6]), *limits[:6].T)
    native_ok, native_peak = inspect(original)
    if native_ok:
        return HomePath(
            start, np.zeros(6), True, 0.0, native_peak, native_peak, float(initial_xyz[0])
        )

    # A small deterministic family changes only relative joint progress.
    # No new joint extrema, arbitrary IK branches, or unbounded online search.
    choices = sorted(product((0.0, -0.75, 0.75), repeat=4), key=lambda v: sum(abs(x) for x in v))
    samples = np.linspace(0, 1, max(129, math.ceil(np.max(np.abs(start)) * 7) + 1))
    for pan, shoulder, elbow, wrist in choices:
        bend = np.array([pan, shoulder, elbow, wrist, wrist, wrist])
        path = HomePath(start, bend, False, 0.0, native_peak, 0.0, float(initial_xyz[0]))
        valid, peak = inspect(path.joints(samples), keep_height=True)
        if not valid:
            continue
        # Quintic time scaling has max s'=1.875 and max |s''|=10/sqrt(3).
        velocity = np.abs(start) * (1 + np.abs(bend)) * 1.875
        acceleration = np.abs(start) * (
            2 * np.abs(bend) * 1.875**2 + (1 + np.abs(bend)) * 10 / math.sqrt(3)
        )
        duration = max(
            0.25,
            float(np.max(velocity)) / MAX_SPEED_DEG_S,
            math.sqrt(
                float(np.max(acceleration)) / (MAX_ACCEL_DEG_S2 - MAX_SPEED_DEG_S * RATE_SLEW_S)
            ),
        )
        return HomePath(start, bend, False, duration, native_peak, peak, float(initial_xyz[0]))
    raise HardwareError(
        "No continuous inward home found for the 0.50m forward plane; no arm trajectory dispatched"
    )


def execute_home(adapter, codec, path):
    """Stream the complete arm path, then hand off near zero to native home."""
    deadline = time.monotonic() + adapter.shutdown_cfg.home_timeout_s
    previous_active = adapter._home_active
    adapter._home_active = True
    phase = (adapter._home_progress_state or {}).get("phase", "unknown")
    if adapter._home_progress_state is not None:
        adapter._home_progress_state.update(stage="forward_path", emitted=None)
    elapsed = 0.0
    rate = 1.0
    stable_since = None
    last_log = -math.inf
    peak = path.start_x_m
    target = path.start.copy()
    try:
        measured, _ = adapter._home_feedback(refresh=True)
        grip = float(
            np.clip(
                (adapter.last_target if adapter.last_target is not None else measured)[6],
                *adapter.limits[6],
            )
        )
        while True:
            measured, _ = adapter._home_feedback(refresh=True)
            now = time.monotonic()
            x = float(tool_positions(codec, measured[:6])[0])
            peak = max(peak, x)
            if now >= deadline:
                raise HardwareError("Coordinated forward home exceeded the existing homing timeout")
            if (
                elapsed >= path.duration_s
                and np.max(np.abs(measured[:6])) <= adapter.shutdown_cfg.arm_tolerance_deg
            ):
                if stable_since is None:
                    stable_since = now
                if now - stable_since >= adapter.shutdown_cfg.settle_s:
                    # The native finishing motion must also fit, using actual
                    # arrival feedback rather than the ideal endpoint.
                    if plan_home(codec, measured, adapter.limits).native:
                        adapter._event(
                            "home_forward_complete",
                            phase=phase,
                            measured_forward_m=x,
                            peak_measured_forward_m=peak,
                        )
                        return
            else:
                stable_since = None
            # Tracking lag slows progress continuously, without freezing at a
            # workspace boundary or jumping ahead after a delayed control tick.
            lag = float(np.max(np.abs(measured[:6] - target)))
            requested_rate = max(0.2, min(1.0, 2.0 / max(2.0, lag)))
            rate += float(
                np.clip(requested_rate - rate, -PERIOD_S * RATE_SLEW_S, PERIOD_S * RATE_SLEW_S)
            )
            elapsed = min(path.duration_s, elapsed + PERIOD_S * rate)
            u = elapsed / path.duration_s
            progress = u**3 * (10 + u * (-15 + 6 * u))
            target = path.joints(progress)
            adapter._send(np.r_[target, grip] / adapter.directions)
            if now - last_log >= 1.0:
                adapter._event(
                    "home_forward_progress",
                    phase=phase,
                    measured_forward_m=x,
                    target_forward_m=float(tool_positions(codec, target)[0]),
                    peak_measured_forward_m=peak,
                    limit_m=FORWARD_LIMIT_M,
                    progress=float(progress),
                    tracking_lag_deg=lag,
                )
                logging.getLogger(__name__).info(
                    "Homing %s/forward_path: x=%.3fm, limit=%.2fm, progress=%.1f%%",
                    phase,
                    x,
                    FORWARD_LIMIT_M,
                    100 * progress,
                )
                last_log = now
            time.sleep(PERIOD_S)
    finally:
        adapter._home_active = previous_active
