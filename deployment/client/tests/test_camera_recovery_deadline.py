"""Reproduce USB warmup crossing the old pair timeout, using virtual time."""

import threading
from collections import deque
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest

from wallx_client.camera_recovery import CameraRecovery, CameraRecoveryBudget
from wallx_client.contracts import CameraUnavailable, HardwareError
from wallx_client.hardware.b601 import Cameras
from wallx_client.hardware.camera_metrics import CameraMetrics


def test_only_first_reopen_can_extend_an_outage():
    budget = CameraRecoveryBudget(1)
    assert budget.wait(10) == 11
    assert budget.reopen(10.15) == 11.15
    assert budget.reopen(10.8) == 11.15
    assert budget.wait(11.1) == 11.15
    with pytest.raises(HardwareError, match="recovery timeout"):
        budget.reopen(11.15)
    budget.clear()  # Only publishing a fresh synchronized pair resets the budget.
    assert budget.wait(12) == 13


def test_late_reopen_cannot_revive_expired_gap():
    budget = CameraRecoveryBudget(1)
    budget.wait(10)
    with pytest.raises(HardwareError, match="recovery timeout"):
        budget.reopen(11)
    assert budget.deadline == 11


def test_control_gate_caps_repeated_deadline_extensions(monkeypatch):
    now = [10.0]
    monkeypatch.setattr("wallx_client.camera_recovery.time.monotonic", lambda: now[0])
    gate = CameraRecovery(1, lambda *a, **kw: None)

    def missing():
        raise CameraUnavailable("reopening", recovery_deadline=now[0] + 1)

    assert gate.poll(missing) is None
    now[0] = 11.5
    assert gate.poll(missing) is None
    now[0] = 12
    with pytest.raises(HardwareError, match="recovery timed out"):
        gate.poll(missing)


def test_control_tick_after_capture_deadline_accepts_pair_published_in_time(monkeypatch):
    now = [10.0]
    monkeypatch.setattr("wallx_client.camera_recovery.time.monotonic", lambda: now[0])
    gate = CameraRecovery(1, lambda *a, **kw: None)

    def missing():
        raise CameraUnavailable("reopening", recovery_deadline=11.15)

    gate.poll(lambda: SimpleNamespace(camera_generation=0))
    assert gate.poll(missing) is None
    # The capture worker publishes at 11.14, then the control tick runs at 11.16.
    now[0] = 11.16
    assert gate.poll(lambda: SimpleNamespace(camera_generation=1)) is not None
    assert gate.recovered

    # A later ordinary gap still enforces the original timeout.
    def ordinary_gap():
        raise CameraUnavailable("late frame")

    gate.poll(ordinary_gap)
    now[0] += 1.1
    with pytest.raises(HardwareError, match="recovery timed out"):
        gate.poll(lambda: "too late")


@pytest.mark.parametrize("recovers", [True, False])
def test_pair_and_control_share_reopen_deadline_from_experiment(cfg, monkeypatch, recovers):
    # 16:27 run: gap -> reopen +149 ms -> old startup frames +~1 s ->
    # first fresh pair +1.025 s. The previous pair timeout fired at +1 s.
    now = [10.0]
    monkeypatch.setattr("wallx_client.hardware.b601.time.monotonic", lambda: now[0])
    camera = Cameras.__new__(Cameras)
    camera.cfg = replace(cfg.robot, camera_recovery_timeout_s=1)
    camera.recovery = CameraRecoveryBudget(1)
    camera.lock = threading.Lock()
    camera.error = None
    camera.generation = camera.latest_generation = 0
    camera.reopening = {}
    camera.reopen_counts = {"side": 0}
    camera.buffers = {name: deque() for name in ("front", "side")}
    image = np.zeros((2, 2, 3), np.uint8)
    camera.latest = ({"front": image, "side": image}, 9.8, 0, 0, 0)
    cap = SimpleNamespace(release=lambda: None)
    camera.devices = {"side": cap}
    camera.formats = {}
    camera._open_camera = lambda name: (cap, {})
    events = []
    camera._report = events.append
    gate = CameraRecovery(1, lambda *a, **kw: None)
    ticks = iter([10.149, 10.497, 10.997, 11.005, 11.025, 11.151])
    done = []

    def wait(_):
        if camera.latest is not None and camera.latest_generation == 1:
            assert gate.poll(camera.snapshot) is not None
            done.append(True)
            return
        now[0] = next(ticks)
        if now[0] == 10.149:
            camera._reopen_camera("side", cap, "USB disconnect")
        if now[0] == 10.997:
            for frames in camera.buffers.values():
                frames.append((1, 10.497, now[0], 0, image))
        if recovers and now[0] == 11.025:
            # Reader has drained startup frames; preserve their original age.
            camera.reopening.clear()
            for frames in camera.buffers.values():
                frames.append((2, 10.94, now[0], 0, image))
        assert gate.poll(camera.snapshot) is None

    camera.stop = SimpleNamespace(is_set=lambda: bool(done), wait=wait)
    assert gate.poll(camera.snapshot) is None
    camera._capture()
    if recovers:
        assert camera.error is None
        assert gate.recovered
        assert gate.max_wait_s > 1
        assert camera.latest[1] == 10.94
        assert "camera_pair_recovered" in events
        assert camera.recovery.deadline is None
    else:
        assert "recovery timeout" in str(camera.error)
        assert camera.latest is None
        assert now[0] == 11.151


def test_reopened_stream_stays_pending_until_first_fresh_frame(cfg, monkeypatch):
    now = [10.0]
    monkeypatch.setattr("wallx_client.hardware.b601.time.monotonic", lambda: now[0])
    camera = Cameras.__new__(Cameras)
    camera.cfg = cfg.robot
    camera.lock = threading.Lock()
    camera.stop = threading.Event()
    camera.error = None
    camera.generation = 1
    camera.reopening = {"side": 11}
    camera.metrics = {"side": CameraMetrics()}
    camera.buffers = {"side": deque()}
    grabs = []

    def grab():
        if grabs:
            assert "side" in camera.reopening  # First two conservative bounds are old.
        grabs.append(True)
        now[0] = [10.5, 10.53, 10.56][len(grabs) - 1]
        if len(grabs) == 3:
            camera.stop.set()
        return True

    cap = SimpleNamespace(
        grab=grab,
        retrieve=lambda: (True, np.zeros((480, 640, 3), np.uint8)),
        release=lambda: None,
    )
    camera.devices = {"side": cap}
    camera._read_camera("side", cap)
    assert camera.error is None
    assert not camera.reopening
    assert [f[1] for f in camera.buffers["side"]] == [10, 10, 10.5]
