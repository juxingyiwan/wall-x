"""Queued prediction IO must drain and report errors, without blocking inference."""

import threading
import time

import numpy as np
import pytest

from wallx_client.contracts import Prediction
from wallx_client.recording import Recorder


def test_queued_prediction_failure_is_reported(cfg, observation):
    recorder = Recorder(cfg, "recording_failure")
    failed = threading.Event()

    def fail(result):
        failed.set()
        raise OSError("simulated disk failure")

    recorder.prediction = fail
    result = Prediction("failed", 0, np.zeros((32, 7)), observation, time.monotonic(), 0, {})
    recorder.prediction_async(result)
    assert failed.wait(2)
    recorder.thread.join(timeout=2)
    with pytest.raises(OSError, match="simulated disk failure"):
        recorder.prediction_async(result)
    with pytest.raises(OSError, match="simulated disk failure"):
        recorder.close({"outcome": "fault"})
