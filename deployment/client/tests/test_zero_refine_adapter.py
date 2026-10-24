"""Exercise the real homing wait loop against load and hard-stop models."""

import math
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest
from test_shutdown import fake_arm

import wallx_client.hardware.b601 as adapter
import wallx_client.hardware.zero_servo as zero


@pytest.mark.parametrize("blocked", [False, True])
def test_homing_loop_converges_or_reports_blockage_and_removes_compensation(
    cfg, monkeypatch, blocked
):
    position = np.array([0.0, 0, 0, 0, 0, 0, 2.483])
    hw, history = fake_arm(cfg, position=position)
    hw.shutdown_cfg = replace(hw.shutdown_cfg, home_timeout_s=10.0)
    clock = SimpleNamespace(now=100.0)

    def sleep(dt):
        clock.now += dt

    timer = SimpleNamespace(monotonic=lambda: clock.now, time=lambda: clock.now, sleep=sleep)
    monkeypatch.setattr(adapter, "time", timer)
    monkeypatch.setattr(zero, "time", timer)
    state = SimpleNamespace(pos=math.radians(position[6]), vel=0.0)
    hw.robot.config = SimpleNamespace(
        gripper_mit_hold_torque_limit=1.0, gripper_mit_torque_limit=3.5
    )
    motor = hw.robot.motors["gripper"]
    motor.get_state = lambda: state
    load = 12 * state.pos
    commands = []

    def send(target):
        assert np.array_equal(target, np.zeros(7))
        native = max(-1.0, min(1.0, -12 * state.pos - 0.05 * state.vel))
        torque = zero.corrected_gripper_torque(hw.robot, motor, 0.0, native)
        commands.append(torque)
        state.vel = (torque + load) / 1.5
        state.pos += state.vel / 30
        if blocked and state.pos < math.radians(2.483):
            state.pos, state.vel = math.radians(2.483), 0.0
        position[6] = math.degrees(state.pos)

    hw._send = send
    if blocked:
        with pytest.raises(adapter.HardwareError, match="gripper zero stalled") as error:
            hw._home(hw.control_cfg)
        assert error.value.homing_context["refinement"]["arrival"]["stalled"]
        assert clock.now < 103
        with pytest.raises(adapter.HardwareError, match="Homing inhibited"):
            hw.close()
        assert sum(value.startswith("disable:") for value in history) == 7
    else:
        measured = hw._home(hw.control_cfg)
        assert abs(measured[6]) <= zero.ZERO_PRECISION_DEG
        assert hw._last_zero_refinement["arrival"]["stationary"]
    assert history.count("home") == 1
    assert max(map(abs, commands)) <= 1.0
    assert not hasattr(hw.robot, "_homing_zero_servo")
