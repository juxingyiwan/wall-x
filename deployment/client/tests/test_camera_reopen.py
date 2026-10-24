"""Owner-only reconnect, bounded retries, stream invalidation and clean cancellation."""

import os
import threading
import time
from collections import deque
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest

from wallx_client.camera_recovery import CameraRecoveryBudget
from wallx_client.config import validate_config
from wallx_client.contracts import HardwareError, Prediction
from wallx_client.controller import check_prediction
from wallx_client.hardware.b601 import Cameras
from wallx_client.hardware.camera_metrics import CameraMetrics


def reader(cfg):
    camera = Cameras.__new__(Cameras)
    camera.cfg = replace(cfg.robot, camera_recovery_timeout_s=0.15)
    camera.recovery = CameraRecoveryBudget(getattr(camera.cfg, "camera_recovery_timeout_s", 1))
    camera.lock = threading.Lock()
    camera.stop = threading.Event()
    camera.error = None
    camera.generation = camera.latest_generation = 0
    camera.reopening = {}
    camera.reopen_counts = {"wrist": 0}
    camera.buffers = {"wrist": deque(), "front": deque([(0, 0, 0, 0, None)])}
    camera.devices = {}
    camera.latest = "old pair"
    camera.metrics = {"wrist": CameraMetrics()}
    camera.formats = {}
    camera._report = lambda kind: None
    return camera


def test_reopen_releases_old_handle_on_owner_and_clears_both_streams(cfg):
    camera = reader(cfg)
    released = []
    old = SimpleNamespace(
        grab=lambda: False, release=lambda: released.append(("old", threading.get_ident()))
    )
    camera.devices["wrist"] = old

    def retrieve():
        camera.stop.set()
        return True, np.zeros((480, 640, 3), dtype=np.uint8)

    new = SimpleNamespace(
        grab=lambda: True,
        retrieve=retrieve,
        release=lambda: released.append(("new", threading.get_ident())),
    )
    camera._open_camera = lambda name: (new, {"checked": True})
    owner = threading.get_ident()
    camera._read_camera("wrist", old)
    assert camera.error is None
    assert camera.generation == 1
    assert camera.latest is None
    assert not camera.buffers["front"]
    assert camera.reopen_counts == {"wrist": 1}
    assert not camera.reopening
    assert released == [("old", owner), ("new", owner)]
    assert not camera.devices


def test_reopen_failure_has_deadline_and_releases_old_handle(cfg):
    camera = reader(cfg)
    released = []
    old = SimpleNamespace(release=lambda: released.append(True))
    camera.devices["wrist"] = old

    def unavailable(name):
        raise HardwareError("device missing")

    camera._open_camera = unavailable
    started = time.monotonic()
    with pytest.raises(HardwareError, match="reopen timed out"):
        camera._reopen_camera("wrist", old, "dequeue failed")
    assert 0.14 <= time.monotonic() - started < 0.5
    assert released == [True]
    assert not camera.devices


def test_inflight_frame_from_pre_reset_generation_is_discarded(cfg):
    camera = reader(cfg)
    camera.buffers["wrist"].clear()

    def retrieve():
        camera.generation += 1  # Other capture worker reset the stream during grab.
        camera.stop.set()
        return True, np.zeros((480, 640, 3), dtype=np.uint8)

    cap = SimpleNamespace(grab=lambda: True, retrieve=retrieve, release=lambda: None)
    camera.devices["wrist"] = cap
    camera._read_camera("wrist", cap)
    assert camera.error is None
    assert not camera.buffers["wrist"]


def test_reopened_handles_without_valid_frames_exhaust_retry_limit(cfg):
    camera = reader(cfg)
    old = SimpleNamespace(grab=lambda: False, release=lambda: None)
    new = SimpleNamespace(grab=lambda: False, release=lambda: None)
    camera.devices["wrist"] = old
    camera._open_camera = lambda name: (new, {})
    camera._read_camera("wrist", old)
    assert "reopen limit" in str(camera.error)
    assert camera.reopen_counts["wrist"] == camera.cfg.camera_max_reopens
    assert not camera.devices


def test_reopen_cancel_during_open_releases_new_handle(cfg):
    camera = reader(cfg)
    released = []
    old = SimpleNamespace(release=lambda: released.append("old"))
    new = SimpleNamespace(release=lambda: released.append("new"))
    camera.devices["wrist"] = old

    def opening(name):
        camera.stop.set()
        return new, {}

    camera._open_camera = opening
    assert camera._reopen_camera("wrist", old, "gap") is None
    assert released == ["old", "new"]
    assert not camera.devices


def test_reopen_does_not_reset_limit_when_it_succeeds(cfg):
    camera = reader(cfg)
    cap = SimpleNamespace(release=lambda: None)
    camera.devices["wrist"] = cap
    camera._open_camera = lambda name: (cap, {})
    for _ in range(camera.cfg.camera_max_reopens):
        assert camera._reopen_camera("wrist", cap, "gap") is cap
    with pytest.raises(HardwareError, match="reopen limit"):
        camera._reopen_camera("wrist", cap, "gap")


def test_close_never_releases_a_handle_owned_by_blocked_reader(cfg):
    camera = reader(cfg)
    released = []
    camera.devices["wrist"] = SimpleNamespace(release=lambda: released.append(True))
    camera.thread = SimpleNamespace(join=lambda **kw: None)
    camera.workers = [SimpleNamespace(join=lambda **kw: None, is_alive=lambda: True)]
    with pytest.raises(HardwareError, match="did not stop"):
        camera.close()
    assert camera.stop.is_set()
    assert not released


def test_preflight_rejects_prediction_from_before_stream_reset(cfg, observation):
    prediction = Prediction("old", 0, np.zeros((32, 7)), observation, time.monotonic(), 0.01, {})
    with pytest.raises(ValueError, match="camera stream generation"):
        check_prediction(prediction, replace(observation, camera_generation=1), cfg.control, 0)


@pytest.mark.parametrize("value", [-1, 11, True, 1.5])
def test_invalid_reopen_count(cfg, value):
    with pytest.raises(ValueError, match="camera_max_reopens"):
        validate_config(replace(cfg, robot=replace(cfg.robot, camera_max_reopens=value)))


@pytest.mark.parametrize(
    "field,value",
    [("width", 320), ("height", 240), ("fourcc", "YUYV"), ("fps", 15), ("buffers", 1)],
)
def test_reopen_verifies_capture_format(cfg, monkeypatch, field, value):
    import cv2

    values = dict(width=640, height=480, fourcc="MJPG", fps=30, buffers=2)
    values[field] = value
    props = {
        cv2.CAP_PROP_FRAME_WIDTH: values["width"],
        cv2.CAP_PROP_FRAME_HEIGHT: values["height"],
        cv2.CAP_PROP_FOURCC: cv2.VideoWriter_fourcc(*values["fourcc"]),
        cv2.CAP_PROP_FPS: values["fps"],
        cv2.CAP_PROP_BUFFERSIZE: values["buffers"],
    }
    released = []
    cap = SimpleNamespace(
        isOpened=lambda: True,
        set=lambda *a: True,
        get=lambda p: props[p],
        release=lambda: released.append(True),
    )
    monkeypatch.setattr(cv2, "VideoCapture", lambda *a: cap)
    camera = reader(cfg)
    camera.cfg = replace(cfg.robot, camera_devices={"wrist": "/fake/by-id/wrist"})
    with pytest.raises(HardwareError, match="readback mismatch"):
        camera._open_camera("wrist")
    assert released == [True]


def test_v4l_select_timeout_is_set_before_opening_first_handle(cfg, monkeypatch):
    import cv2

    camera = reader(cfg)
    camera.cfg = replace(cfg.robot, camera_devices={"wrist": "/fake/by-id/wrist"})
    monkeypatch.setenv("OPENCV_VIDEOIO_V4L_SELECT_TIMEOUT", "10")
    props = {
        cv2.CAP_PROP_FRAME_WIDTH: 640,
        cv2.CAP_PROP_FRAME_HEIGHT: 480,
        cv2.CAP_PROP_FOURCC: cv2.VideoWriter_fourcc(*"MJPG"),
        cv2.CAP_PROP_FPS: 30,
        cv2.CAP_PROP_BUFFERSIZE: 2,
    }
    cap = SimpleNamespace(isOpened=lambda: True, set=lambda *a: True, get=props.get)

    def opening(*args):
        assert os.environ["OPENCV_VIDEOIO_V4L_SELECT_TIMEOUT"] == "1"
        return cap

    monkeypatch.setattr(cv2, "VideoCapture", opening)
    result, fmt = camera._open_camera("wrist")
    assert result is cap
    assert fmt["select_timeout_s"] == 1


class VirtualStop:
    def __init__(self, now):
        self.now = now
        self.stopped = False

    def is_set(self):
        return self.stopped

    def set(self):
        self.stopped = True

    def wait(self, duration):
        self.now[0] += duration
        return self.stopped


@pytest.mark.parametrize("failure", ["false", "cv2_error"])
@pytest.mark.parametrize("reopens", [0, 3])
def test_one_slow_failed_read_reaches_reopen_before_three_second_deadline(
    cfg, monkeypatch, failure, reopens
):
    import cv2

    now = [10.0]
    monkeypatch.setattr("wallx_client.hardware.b601.time.monotonic", lambda: now[0])
    camera = reader(cfg)
    camera.cfg = replace(cfg.robot, camera_recovery_timeout_s=3, camera_max_reopens=reopens)
    camera.recovery = CameraRecoveryBudget(3)
    camera.recovery.wait(now[0])
    camera.stop = VirtualStop(now)
    events = []
    camera._report = events.append
    released = []
    attempts = []
    owner = threading.get_ident()

    def blocked():
        attempts.append(now[0])
        now[0] += 1.001  # OpenCV's one-second select wait returns failure.
        if failure == "cv2_error":
            raise cv2.error("dequeue timeout")
        return False

    old = SimpleNamespace(
        grab=blocked, release=lambda: released.append(("old", threading.get_ident()))
    )
    camera.devices["wrist"] = old

    def retrieve():
        camera.stop.set()
        return True, np.zeros((480, 640, 3), dtype=np.uint8)

    new = SimpleNamespace(
        grab=lambda: True,
        retrieve=retrieve,
        release=lambda: released.append(("new", threading.get_ident())),
    )
    camera._open_camera = lambda name: (new, {})
    camera._read_camera("wrist", old)
    assert attempts == [10.0]
    assert events[0] == "camera_read_timeout"
    assert not camera.devices
    if reopens:
        assert camera.error is None
        assert camera.reopen_counts["wrist"] == 1
        assert camera.generation == 1
        assert camera.latest is None
        assert not camera.buffers["front"]
        assert camera.recovery.deadline == pytest.approx(14.001)
        assert released == [("old", owner), ("new", owner)]
    else:
        assert "capture failures" in str(camera.error)
        assert camera.reopen_counts["wrist"] == 0
        assert released == [("old", owner)]


def test_successful_slow_startup_read_keeps_conservative_timestamp(cfg, monkeypatch):
    now = [10.0]
    monkeypatch.setattr("wallx_client.hardware.b601.time.monotonic", lambda: now[0])
    camera = reader(cfg)
    camera.stop = VirtualStop(now)
    events = []
    camera._report = events.append

    def slow_grab():
        now[0] += 1.1
        return True

    def retrieve():
        camera.stop.set()
        return True, np.zeros((480, 640, 3), dtype=np.uint8)

    cap = SimpleNamespace(grab=slow_grab, retrieve=retrieve, release=lambda: None)
    camera.devices["wrist"] = cap
    camera._read_camera("wrist", cap)
    assert camera.error is None
    assert camera.reopen_counts["wrist"] == 0
    assert not events
    assert camera.buffers["wrist"][0][1] == 10
    assert camera._select_pair(camera.buffers, {"wrist": -1, "front": -1}, now[0]) is None


def test_reopened_handle_that_times_out_cannot_restart_recovery_window(cfg, monkeypatch):
    now = [10.0]
    monkeypatch.setattr("wallx_client.hardware.b601.time.monotonic", lambda: now[0])
    camera = reader(cfg)
    camera.cfg = replace(cfg.robot, camera_recovery_timeout_s=3, camera_max_reopens=10)
    camera.recovery = CameraRecoveryBudget(3)
    camera.recovery.wait(now[0])
    camera.stop = VirtualStop(now)
    released = []

    def timeout():
        now[0] += 1.001
        return False

    old = SimpleNamespace(grab=timeout, release=lambda: released.append("old"))
    new = SimpleNamespace(grab=timeout, release=lambda: released.append("new"))
    camera.devices["wrist"] = old
    camera._open_camera = lambda name: (new, {})
    camera._read_camera("wrist", old)
    assert "recovery timeout" in str(camera.error)
    assert camera.reopen_counts["wrist"] == 3
    assert camera.recovery.deadline == pytest.approx(14.001)
    assert now[0] == pytest.approx(14.004)
    assert released == ["old", "new", "new", "new"]
    assert not camera.devices


@pytest.mark.parametrize("failure", ["grab", "frame", "cv2_error"])
def test_second_disconnect_during_reopen_can_recover_within_same_deadline(
    cfg, monkeypatch, failure
):
    """USB re-enumeration can invalidate a newly opened, not-yet-streaming handle."""
    import cv2

    now = [10.0]
    monkeypatch.setattr("wallx_client.hardware.b601.time.monotonic", lambda: now[0])
    camera = reader(cfg)
    camera.cfg = replace(cfg.robot, camera_recovery_timeout_s=3)
    camera.recovery = CameraRecoveryBudget(3)
    camera.stop = VirtualStop(now)
    released, deadlines, generations = [], [], []
    owner = threading.get_ident()

    def failed_grab():
        now[0] += 0.04
        if failure == "cv2_error":
            raise cv2.error("device disappeared again")
        return failure == "frame"

    old = SimpleNamespace(
        grab=lambda: False, release=lambda: released.append(("old", threading.get_ident()))
    )
    bad = SimpleNamespace(
        grab=failed_grab,
        retrieve=lambda: (False, None),
        release=lambda: released.append(("bad", threading.get_ident())),
    )

    def good_frame():
        camera.stop.set()
        return True, np.zeros((480, 640, 3), dtype=np.uint8)

    good = SimpleNamespace(
        grab=lambda: True,
        retrieve=good_frame,
        release=lambda: released.append(("good", threading.get_ident())),
    )
    handles = iter((bad, good))

    def opening(name):
        deadlines.append(camera.recovery.deadline)
        generations.append(camera.generation)
        return next(handles), {}

    camera.devices["wrist"] = old
    camera._open_camera = opening
    camera._read_camera("wrist", old)
    assert camera.error is None
    assert len(deadlines) == 2 and deadlines[0] == deadlines[1]
    assert now[0] < deadlines[0]
    assert generations == [1, 2]
    assert camera.reopen_counts["wrist"] == 2
    assert not camera.reopening
    assert camera.latest is None  # Only the pair owner can publish the new stream.
    assert not camera.buffers["front"]
    assert len(camera.buffers["wrist"]) == 1
    assert released == [("old", owner), ("bad", owner), ("good", owner)]
    assert not camera.devices
