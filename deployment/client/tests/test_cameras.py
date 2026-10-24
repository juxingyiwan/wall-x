"""Pair independent camera streams without relaxing freshness or skew limits."""

import threading
from types import SimpleNamespace

import numpy as np
import pytest

from wallx_client.camera_recovery import CameraRecoveryBudget
from wallx_client.contracts import HardwareError
from wallx_client.hardware.b601 import Cameras


def worker():
    camera = Cameras.__new__(Cameras)
    camera.cfg = SimpleNamespace(max_camera_skew_s=0.05, max_camera_age_s=0.15)
    camera.latest = camera.error = None
    camera.recovery = CameraRecoveryBudget(getattr(camera.cfg, "camera_recovery_timeout_s", 1))
    camera.lock = threading.Lock()
    return camera


def frame(sequence, started, grabbed):
    return sequence, started, grabbed, 0, np.zeros((2, 2, 3), np.uint8)


def test_fast_camera_matches_slow_camera_newest_frame():
    camera = worker()
    buffers = {
        "front": [frame(0, 0, 0.01), frame(1, 0.033, 0.044), frame(2, 0.066, 0.077)],
        "side": [frame(0, 0, 0.073)],
    }
    pair = camera._select_pair(buffers, {"front": -1, "side": -1}, 0.08)
    assert pair[0][0] == 2
    assert pair[1][0] == 0


def test_consumed_frame_cannot_be_reused_as_new_observation():
    camera = worker()
    assert (
        camera._select_pair(
            {"front": [frame(1, 0.03, 0.04)], "side": [frame(0, 0.01, 0.02)]},
            {"front": 0, "side": 0},
            0.05,
        )
        is None
    )


@pytest.mark.parametrize("now,side_grab", [(0.20, 0.02), (0.10, 0.09)])
def test_stale_or_unsynchronized_pair_rejected(now, side_grab):
    camera = worker()
    assert (
        camera._select_pair(
            {"front": [frame(0, 0, 0.01)], "side": [frame(0, 0, side_grab)]},
            {"front": -1, "side": -1},
            now,
        )
        is None
    )


def test_snapshot_fails_closed_on_worker_error():
    camera = worker()
    camera.error = HardwareError("camera disconnected")
    with pytest.raises(HardwareError, match="disconnected"):
        camera.snapshot()


def test_snapshot_keeps_original_acquisition_start_for_age(monkeypatch):
    camera = worker()
    camera.latest = ({"front": np.zeros((2, 2, 3))}, 10, 0, 0.01, 5)
    monkeypatch.setattr("wallx_client.hardware.b601.time.monotonic", lambda: 10.16)
    with pytest.raises(HardwareError, match="stale"):
        camera.snapshot()
