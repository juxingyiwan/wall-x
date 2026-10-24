"""Camera-only diagnostics must never construct a robot or inference connection."""

import threading
import time
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest

from wallx_client.camera_diagnostics import camera_test
from wallx_client.camera_recovery import CameraRecoveryBudget
from wallx_client.hardware.b601 import Cameras
from wallx_client.hardware.camera_metrics import CameraMetrics, CaptureClock


def test_metrics_distinguish_blocking_capture_from_decode():
    metrics = CameraMetrics()
    assert metrics.summary(10)["measured_fps"] is None
    metrics.completed(10, 10.05, 10.052)
    metrics.completed(10.06, 10.15, 10.152)
    metrics.reading_since = 10.16
    result = metrics.summary(10.2)
    assert result["frames"] == 2
    assert result["measured_fps"] == pytest.approx(10)
    assert result["grab_mean_s"] == pytest.approx(0.07)
    assert result["decode_max_s"] == pytest.approx(0.002)
    assert result["current_read_elapsed_s"] == pytest.approx(0.04)


@pytest.mark.parametrize("fault", ["no_frame", "expired", "skew"])
def test_pair_diagnostics_identify_the_blocked_camera(monkeypatch, fault):
    camera = Cameras.__new__(Cameras)
    camera.cfg = SimpleNamespace(max_camera_age_s=0.15, max_camera_skew_s=0.05)
    camera.recovery = CameraRecoveryBudget(getattr(camera.cfg, "camera_recovery_timeout_s", 1))
    camera.lock = threading.Lock()
    camera.latest = camera.error = None
    camera.generation = 0
    camera.reopen_counts = {}
    camera.reopening = {}
    camera.metrics = {k: CameraMetrics() for k in ("front", "wrist")}
    camera.formats = {k: {} for k in camera.metrics}
    camera.consumed = dict.fromkeys(camera.metrics, -1)

    def frame(start, end):
        return 0, start, end, 0, np.zeros((2, 2, 3))

    camera.buffers = {
        "front": [frame(9.9, 9.91)],
        "wrist": [] if fault == "no_frame" else [frame(9 if fault == "expired" else 9.97, 9.98)],
    }
    monkeypatch.setattr("wallx_client.hardware.b601.time.monotonic", lambda: 10)
    result = camera.diagnostics()
    assert result["next_pair_blockers"] == [
        {
            "no_frame": "no_new_frame:wrist",
            "expired": "expired_frames:wrist",
            "skew": "pair_skew_exceeded",
        }[fault]
    ]


def test_camera_only_command_skips_robot_policy_and_telemetry(cfg, tmp_path, monkeypatch):
    import wallx_client.camera_diagnostics as diagnostics

    closed = []
    camera = SimpleNamespace(
        latest=True,
        error=None,
        snapshot=lambda: ({}, time.monotonic(), time.time(), 0, 0),
        diagnostics=lambda: {"cameras": {}},
        close=lambda: closed.append(True),
    )
    monkeypatch.setattr(diagnostics, "Cameras", lambda *a, **k: camera)
    cfg = replace(cfg, robot=replace(cfg.robot, lock_paths=(str(tmp_path / "camera.lock"),)))
    report = camera_test(cfg, 0.01, SimpleNamespace(event=lambda *a, **k: None))
    assert report["outcome"] == "camera_test_complete"
    assert report["can_opened"] is report["motors_enabled"] is report["policy_connected"] is False
    assert closed == [True]


@pytest.mark.parametrize("duration", [0, -1, float("inf"), float("nan")])
def test_invalid_duration_never_opens_camera(cfg, duration, monkeypatch):
    monkeypatch.setattr(
        "wallx_client.camera_diagnostics.Cameras", lambda *a, **k: pytest.fail("opened")
    )
    with pytest.raises(ValueError, match="finite and positive"):
        camera_test(cfg, duration, object())


@pytest.mark.parametrize("buffers", [1, 2])
def test_queued_images_keep_their_age_after_a_reader_stall(buffers):
    clock = CaptureClock(buffers)
    starts = [10, 10.03, 10.06, 10.50]
    captured = []
    for t in starts:
        captured.append(clock.start(t))
        clock.completed(t + 0.01)
    assert captured[-1] == pytest.approx(starts[-1 - buffers] + 0.01)
    # A newly dequeued but old frame must fail the existing 150 ms age gate.
    assert starts[-1] - captured[-1] > 0.15


def test_blocking_stream_start_does_not_age_subsequent_refilled_buffers():
    clock = CaptureClock(2)
    assert clock.start(10) == 10
    clock.completed(10.50)
    assert clock.start(10.51) == 10
    clock.completed(10.533)
    # The third image reused a buffer returned after the first long grab.
    # Its bound must remain earlier than capture, without including startup.
    assert clock.start(10.54) == 10.50
    clock.completed(10.566)
    assert clock.start(10.58) == 10.533


@pytest.mark.parametrize("buffers", [0, 3, True, 1.5])
def test_unsupported_camera_buffer_count_rejected(cfg, buffers):
    from wallx_client.config import validate_config

    with pytest.raises(ValueError, match="camera_buffer_size"):
        validate_config(replace(cfg, robot=replace(cfg.robot, camera_buffer_size=buffers)))


def test_camera_open_rejects_ignored_buffer_count_before_starting_threads(cfg, monkeypatch):
    import cv2

    released = []
    cap = SimpleNamespace(
        set=lambda *a: True,
        isOpened=lambda: True,
        get=lambda prop: 1 if prop == cv2.CAP_PROP_BUFFERSIZE else 0,
        release=lambda: released.append(True),
    )
    monkeypatch.setattr(cv2, "VideoCapture", lambda *a: cap)
    with pytest.raises(Exception, match="buffer count readback mismatch"):
        Cameras(cfg.robot)
    assert released == [True]


def test_camera_cli_never_opens_robot_policy_or_display(cfg, tmp_path, monkeypatch):
    import yaml

    from wallx_client.cli import main

    def forbidden(*args, **kwargs):
        pytest.fail("camera-test must not initialize robot, policy or telemetry")

    monkeypatch.setattr("wallx_client.hardware.b601.B601", forbidden)
    monkeypatch.setattr("wallx_client.cli.WallXConnection", forbidden)
    monkeypatch.setattr("wallx_client.recording.Recorder.start_telemetry", forbidden)
    camera = SimpleNamespace(
        latest=True,
        error=None,
        snapshot=lambda: ({}, time.monotonic(), time.time(), 0, 0),
        diagnostics=lambda: {},
        close=lambda: None,
    )
    monkeypatch.setattr("wallx_client.camera_diagnostics.Cameras", lambda *a, **k: camera)
    cfg = replace(cfg, robot=replace(cfg.robot, lock_paths=(str(tmp_path / "test.lock"),)))
    path = tmp_path / "config.yml"
    path.write_text(yaml.safe_dump(cfg.to_dict()))
    assert main(["--config", str(path), "camera-test", "--duration", "0.01"]) == 0
