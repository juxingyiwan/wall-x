"""Holding must preserve impedance targets despite nonzero tracking error."""

import time
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest

from wallx_client.contracts import JOINTS, CameraUnavailable, Prediction
from wallx_client.controller import control_loop
from wallx_client.hardware.b601 import B601


def hardware(cfg):
    hw = B601(cfg.robot, lambda *a, **kw: None)
    hw.directions = np.array([1, 1, -1, -1, -1, 1, 6.0])
    hw.limits = np.array(
        [[-145, 145], [0, 170], [0, 200], [-80, 90], [-90, 90], [-130, 130], [0, 270]]
    )

    def send(action):
        raw = np.array([action[j + ".pos"] for j in JOINTS])
        target = np.clip(raw * hw.directions, *hw.limits.T)
        return dict(zip((j + ".pos" for j in JOINTS), target, strict=True))

    hw.robot = SimpleNamespace(send_action=send)
    return hw


def test_initial_hold_uses_measured_pose_before_any_command(cfg):
    hw = hardware(cfg)
    measured = np.array([12, 90, 40, -20, 10, 30, 60.0])
    np.testing.assert_array_equal(hw.hold_command(measured) * hw.directions, measured)
    assert hw.last_target is None


def test_hold_retains_last_dispatched_target_for_all_seven_joints(cfg):
    hw = hardware(cfg)
    requested = np.array([150, 90, 40, -20, 10, 30, 60.0])
    target = np.array(hw.send(requested / hw.directions)["driver_target_physical"])
    error = np.array([0.3, -3, 2.7, 2.1, -0.2, 0.2, 0.5])
    held = hw.hold_command(target - error)
    np.testing.assert_array_equal(held * hw.directions, target)
    held[:] = 0
    np.testing.assert_array_equal(hw.last_target, target)  # No alias of the cached command.


@pytest.mark.parametrize("transition", ["pause", "camera"])
def test_serial_hold_transitions_preserve_support_target(cfg, observation, tmp_path, transition):
    cfg = replace(
        cfg,
        control=replace(
            cfg.control,
            execution_horizon=1,
            duration_s=2,
        ),
    )
    hw = hardware(cfg)
    start = np.array([10, 90, 40, -20, 0, 30, 60.0])
    target = start + np.array([1, 2, 3, -2, 1, 1, 2])
    error = np.array([0.3, -3, 2.7, 2.1, -0.2, 0.2, 0.5])
    events, commands, calls = [], [], []
    completed = False
    gap_ticks = 0
    key_sent = False

    def state():
        # A loaded impedance-controlled arm can be stationary with a persistent
        # target/feedback offset. Perfect-tracking fakes hide a hold reset bug.
        return target - error if completed else start

    def observe():
        nonlocal gap_ticks
        if transition == "camera" and completed and gap_ticks < 2:
            gap_ticks += 1
            raise CameraUnavailable("transient frame gap")
        now = time.monotonic()
        return replace(observation, state=state(), captured_monotonic=now, state_monotonic=now), {}

    def event(kind, **kw):
        nonlocal completed
        events.append(kind)
        if kind == "chunk_completed":
            completed = True

    def append(kind, **kw):
        if kind == "command":
            commands.append((kw["phase"], kw["driver_target_physical"]))

    def predict(obs, task, generation):
        calls.append(generation)
        source = obs
        actions = np.repeat((target / hw.directions)[None, :], 32, axis=0)
        if len(calls) > 1:
            actions[:] = 999  # Rejected prediction must never be executed.
        return Prediction(str(len(calls)), generation, actions, source, time.monotonic(), 0, {})

    def read_key():
        nonlocal key_sent
        if completed and transition == "pause" and not key_sent:
            key_sent = True
            return "p"
        finished = "paused" in events or "camera_wait_recovered" in events
        return "q" if finished else ""

    hw.observe = observe
    hw.observe_state = lambda: (state(), {})
    hw.close = lambda: None
    recorder = SimpleNamespace(
        path=tmp_path,
        event=event,
        append=append,
        prediction=lambda *a: None,
        prediction_async=lambda *a: None,
        sample=lambda *a, **kw: None,
        check_media=lambda: None,
    )
    report = control_loop(
        hw, SimpleNamespace(predict=predict, close=lambda: None), cfg, recorder, read_key
    )
    assert report["outcome"] == "aborted", report
    assert report["source_actions_completed"] == 1
    first = next(i for i, (phase, _) in enumerate(commands) if phase == "policy")
    assert len(commands) > first + 1
    for _, dispatched in commands[first:]:
        np.testing.assert_allclose(dispatched, target, atol=1e-12, rtol=0)
