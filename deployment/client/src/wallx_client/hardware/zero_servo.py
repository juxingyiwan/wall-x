"""Bounded gripper correction used only while verifying a zero return.

This module performs no I/O and never changes a position target or motor zero.
The existing driver supplies the PD torque and retains all command checks.
"""

import math
import time
from contextlib import contextmanager

ZERO_PRECISION_DEG = 0.5
ZERO_SPEED_DEG_S = 2.0
NO_PROGRESS_S = 2.0
MIN_PROGRESS_DEG = 0.1
MAX_REFINEMENT_S = 10.0


class ZeroGripperServo:
    """PI correction around native PD, with anti-windup and bounded torque."""

    def __init__(self):
        self.integral = 0.0
        self.command_time = None
        self.observation = None
        self.best_error = None
        self.progress_time = None
        self.last = {}

    def correct(self, native, target_rad, position_rad, velocity_rad_s, limit, now):
        values = (native, target_rad, position_rad, velocity_rad_s, limit, now)
        if not all(math.isfinite(v) for v in values) or limit <= 0:
            raise ValueError("Invalid gripper zero-servo feedback or torque limit")
        if target_rad != 0.0:
            raise ValueError("Zero refinement must not change the calibrated zero target")
        error = -position_rad
        dt = None if self.command_time is None else now - self.command_time
        self.command_time = now
        # Do not integrate over a scheduler pause or carry correction across zero.
        if dt is not None and (dt <= 0 or dt > 0.1):
            self.integral = 0.0
        if self.integral * error < 0:
            self.integral = 0.0
        if dt is not None and 0 < dt <= 0.1:
            # Ki=6 Nm/(rad*s), with correction slew at most 0.2 Nm/s.
            increment = max(-0.2 * dt, min(0.2 * dt, 6.0 * error * dt))
            candidate = max(-limit, min(limit, self.integral + increment))
            total = native + candidate
            # Conditional integration: no windup against the existing torque cap.
            if abs(total) <= limit or total * increment < 0:
                self.integral = candidate
        command = max(-limit, min(limit, native + self.integral))
        self.last = dict(
            target_physical_deg=0.0,
            measured_physical_deg=math.degrees(position_rad),
            velocity_deg_s=math.degrees(velocity_rad_s),
            native_torque_nm=native,
            integral_torque_nm=self.integral,
            commanded_torque_nm=command,
            torque_limit_nm=limit,
            control_dt_s=dt,
        )
        return command

    def observe(self, position_deg, now, tolerance):
        if not all(math.isfinite(v) for v in (position_deg, now, tolerance)) or tolerance <= 0:
            raise ValueError("Invalid zero-arrival feedback")
        speed = None
        if self.observation is not None:
            before, stamp = self.observation
            dt = now - stamp
            if 0 < dt <= 0.1:
                speed = abs(position_deg - before) / dt
        self.observation = position_deg, now
        error = abs(position_deg)
        if self.best_error is None or error <= self.best_error - MIN_PROGRESS_DEG:
            self.best_error, self.progress_time = error, now
        stalled = error > tolerance and now - self.progress_time >= NO_PROGRESS_S
        return dict(
            stationary=speed is not None and speed <= ZERO_SPEED_DEG_S,
            stalled=stalled,
            speed_deg_s=speed,
            best_error_deg=self.best_error,
            no_progress_s=now - self.progress_time,
        )


def corrected_gripper_torque(robot, motor, target_rad, native):
    """Called after native PD calculation; inactive policy calls are unchanged."""
    servo = getattr(robot, "_homing_zero_servo", None)
    if servo is None:
        return native
    if motor is not robot.motors.get("gripper"):
        raise ValueError("Zero refinement can only control the gripper")
    state = motor.get_state()  # Existing state only; never add a CAN transaction.
    if state is None or native is None:
        raise ValueError("Missing gripper feedback during zero refinement")
    limits = (
        float(robot.config.gripper_mit_hold_torque_limit),
        float(robot.config.gripper_mit_torque_limit),
    )
    if not all(math.isfinite(v) and v > 0 for v in limits):
        raise ValueError("Invalid native gripper torque limits")
    limit = min(limits)
    return servo.correct(
        float(native),
        float(target_rad),
        float(state.pos),
        float(state.vel),
        float(limit),
        time.monotonic(),
    )


@contextmanager
def zero_gripper_session(robot):
    """Remove all correction on success, fault, alarm, or operator interruption."""
    if getattr(robot, "_homing_zero_servo", None) is not None:
        raise RuntimeError("A gripper zero-refinement session is already active")
    sentinel = object()
    previous = getattr(robot, "_homing_zero_servo", sentinel)
    servo = ZeroGripperServo()
    robot._homing_zero_servo = servo
    try:
        yield servo
    finally:
        if previous is sentinel:
            del robot._homing_zero_servo
        else:
            robot._homing_zero_servo = previous
