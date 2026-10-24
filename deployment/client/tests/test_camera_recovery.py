"""Transient camera gaps never weaken motor checks or revive an old prediction."""

import threading
import time
from collections import deque
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest

from wallx_client.camera_recovery import CameraRecovery, CameraRecoveryBudget
from wallx_client.config import validate_config
from wallx_client.contracts import CameraUnavailable, HardwareError, Prediction
from wallx_client.hardware.b601 import Cameras
from wallx_client.hardware.camera_metrics import CameraMetrics


def test_only_camera_unavailability_is_recoverable(monkeypatch):
    now = [10.0]
    monkeypatch.setattr("wallx_client.camera_recovery.time.monotonic", lambda: now[0])
    events = []
    gate = CameraRecovery(1, lambda kind, **kw: events.append(kind))

    def missing():
        raise CameraUnavailable("late frame")

    assert gate.poll(missing) is None
    assert gate.entered
    now[0] += 0.1
    assert gate.poll(lambda: "fresh") == "fresh"
    assert gate.recovered
    assert gate.summary()["recoveries"] == 1
    assert events == ["camera_wait_started", "camera_wait_recovered"]

    def motor_fault():
        raise HardwareError("CAN stale")

    with pytest.raises(HardwareError, match="CAN stale"):
        gate.poll(motor_fault)


def test_camera_deadline_does_not_accept_a_late_recovery(monkeypatch):
    now = [10.0]
    monkeypatch.setattr("wallx_client.camera_recovery.time.monotonic", lambda: now[0])
    gate = CameraRecovery(1, lambda *a, **kw: None)

    def missing():
        raise CameraUnavailable("late frame")

    gate.poll(missing)
    now[0] += 1.1
    with pytest.raises(HardwareError, match="recovery timed out"):
        gate.poll(lambda: "fresh but too late")


@pytest.mark.parametrize("rtc", [False, True])
@pytest.mark.parametrize("fault", ["transient", "persistent", "motor", "stop", "hidden"])
def test_controller_holds_recovers_and_discards_pre_gap_response(
    cfg, observation, tmp_path, monkeypatch, rtc, fault
):
    from wallx_client.controller import control_loop as serial
    from wallx_client.rtc_queue_controller import control_loop as asynchronous

    cfg = replace(
        cfg,
        robot=replace(cfg.robot, camera_recovery_timeout_s=0.25),
        control=replace(cfg.control, duration_s=2),
    )
    calls, commands, events, closed = [], [], [], []
    request_started = threading.Event()
    gap_ticks = 0
    fresh_after_gap = []

    class Hardware:
        directions = np.ones(7)
        limits = np.array([[-500, 500]] * 7)

        def observe(self):
            nonlocal gap_ticks
            if fault == "hidden" and request_started.is_set():
                gap_ticks = 3
            elif request_started.is_set() and (gap_ticks < 3 or fault == "persistent"):
                gap_ticks += 1
                raise CameraUnavailable("wrist frame late")
            now = time.monotonic()
            if gap_ticks:
                fresh_after_gap.append(now)
            return replace(
                observation,
                state=np.zeros(7),
                captured_monotonic=now,
                state_monotonic=now,
                camera_generation=int(fault == "hidden" and bool(gap_ticks)),
            ), {}

        def observe_state(self):
            if fault == "motor":
                raise HardwareError("CAN failed during camera recovery")
            return np.zeros(7), {}

        def hold_command(self, state):
            return state

        def send(self, raw):
            assert not closed
            commands.append(np.asarray(raw).copy())
            return {"driver_target_physical": np.asarray(raw).tolist()}

        def close(self):
            closed.append("hardware")

    class Connection:
        def predict(self, obs, task, generation, **kw):
            calls.append((generation, obs.captured_monotonic))
            first = len(calls) == 1
            request_started.set()
            if first:
                time.sleep(0.18)  # Delayed pre-gap response must never reach the arm.
            return Prediction(
                str(len(calls)),
                generation,
                np.full((32, 7), 10 if first else 0),
                obs,
                time.monotonic(),
                0.01,
                {},
                {"mode": "bootstrap"},
            )

        def close(self):
            closed.append("network")

    recorder = SimpleNamespace(
        path=tmp_path,
        event=lambda kind, **kw: events.append((kind, kw)),
        append=lambda kind, **kw: events.append((kind, kw)),
        prediction=lambda *a: None,
        prediction_async=lambda *a: None,
        sample=lambda *a, **kw: None,
        check_media=lambda: None,
    )

    def key():
        if fault == "stop" and gap_ticks:
            return "q"
        if any(kind in ("chunk_accepted", "rtc_queue_replaced") for kind, _ in events):
            return "q"
        return ""

    report = (asynchronous if rtc else serial)(Hardware(), Connection(), cfg, recorder, key)
    assert closed == ["hardware", "network"]
    assert all(np.allclose(c, 0) for c in commands)  # The old 10-degree plan was discarded.
    if fault in ("transient", "hidden"):
        assert report["outcome"] == "aborted", report
        assert report["camera_recovery"]["recoveries"] == 1
        assert report["discarded_responses"] == 1
        assert calls[1][0] > calls[0][0]
        assert calls[1][1] >= fresh_after_gap[0]
    elif fault == "stop":
        assert report["outcome"] == "aborted"
        assert report["camera_recovery"]["recoveries"] == 0
    else:
        assert report["outcome"] == "fault"
        assert ("CAN failed" if fault == "motor" else "recovery timed out") in report["error"]
    if fault != "motor":
        assert any(kw.get("phase") == "camera_wait" for _, kw in events)


def test_pair_worker_recovers_without_relaxing_frame_age(cfg):
    camera = Cameras.__new__(Cameras)
    camera.cfg = replace(cfg.robot, max_camera_age_s=0.05, camera_recovery_timeout_s=0.3)
    camera.devices = {"front": None, "wrist": None}
    camera.recovery = CameraRecoveryBudget(getattr(camera.cfg, "camera_recovery_timeout_s", 1))
    camera.lock = threading.Lock()
    camera.stop = threading.Event()
    camera.error = None
    camera.generation = camera.latest_generation = 0
    camera.reopening = {}
    now = time.monotonic()
    image = np.zeros((2, 2, 3), dtype=np.uint8)
    camera.latest = ({"front": image, "wrist": image}, now - 0.1, time.time(), 0, 0)
    camera.buffers = {name: deque() for name in camera.devices}
    camera._report = lambda kind: None
    thread = threading.Thread(target=camera._capture)
    thread.start()
    try:
        with pytest.raises(CameraUnavailable):
            camera.snapshot()
        assert camera.error is None
        time.sleep(0.02)
        with camera.lock:
            for frames in camera.buffers.values():
                frames.append((1, time.monotonic(), time.monotonic(), time.time(), image))
        deadline = time.monotonic() + 0.2
        while time.monotonic() < deadline:
            try:
                value = camera.snapshot()
                break
            except CameraUnavailable:
                time.sleep(0.002)
        else:
            pytest.fail("pair worker did not recover")
        assert time.monotonic() - value[1] <= camera.cfg.max_camera_age_s
        assert camera.error is None
    finally:
        camera.stop.set()
        thread.join(timeout=1)


@pytest.mark.parametrize("failures", [1, 3])
def test_reader_retries_isolated_bad_capture_but_bounds_consecutive_failures(cfg, failures):
    camera = Cameras.__new__(Cameras)
    camera.cfg = replace(cfg.robot, camera_max_reopens=0)
    camera.recovery = CameraRecoveryBudget(getattr(camera.cfg, "camera_recovery_timeout_s", 1))
    camera.lock = threading.Lock()
    camera.stop = threading.Event()
    camera.error = None
    camera.generation = 0
    camera.devices = {}
    camera.reopening = {}
    camera.metrics = {"wrist": CameraMetrics()}
    camera.buffers = {"wrist": deque()}
    calls = []

    def grab():
        calls.append(True)
        return len(calls) > failures

    def retrieve():
        camera.stop.set()
        return True, np.zeros((480, 640, 3), dtype=np.uint8)

    camera._read_camera("wrist", SimpleNamespace(grab=grab, retrieve=retrieve))
    if failures == 1:
        assert camera.error is None
        assert len(camera.buffers["wrist"]) == 1
    else:
        assert "3 consecutive capture failures" in str(camera.error)
        assert not camera.buffers["wrist"]


@pytest.mark.parametrize("value", [0, -1, True, 1.5])
def test_invalid_capture_failure_limit(cfg, value):
    with pytest.raises(ValueError, match="camera_max_consecutive_failures"):
        validate_config(
            replace(cfg, robot=replace(cfg.robot, camera_max_consecutive_failures=value))
        )


@pytest.mark.parametrize("value", [0, -1, True, float("nan"), float("inf")])
def test_invalid_recovery_timeout(cfg, value):
    with pytest.raises(ValueError, match="camera_recovery_timeout_s"):
        validate_config(replace(cfg, robot=replace(cfg.robot, camera_recovery_timeout_s=value)))
