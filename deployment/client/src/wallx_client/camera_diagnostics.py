"""Exercise the production camera pipeline without constructing a robot or policy."""

import math
import time

from .camera_recovery import CameraRecovery
from .hardware.b601 import Cameras
from .locks import robot_locks


def camera_test(cfg, duration, recorder):
    if not math.isfinite(duration) or duration <= 0:
        raise ValueError("camera-test duration must be finite and positive")
    report = {
        "outcome": "running",
        "can_opened": False,
        "motors_enabled": False,
        "policy_connected": False,
        "requested_duration_s": duration,
        "samples": 0,
    }
    cameras = None
    recovery = CameraRecovery(cfg.robot.camera_recovery_timeout_s, recorder.event)
    with robot_locks(cfg.robot.lock_paths):
        try:
            cameras = Cameras(cfg.robot, event=recorder.event)
            warmup = time.monotonic() + 3
            while cameras.latest is None and cameras.error is None and time.monotonic() < warmup:
                time.sleep(0.01)
            cameras.snapshot()
            started = time.monotonic()
            report["started_wall"] = time.time()
            age_total = 0.0
            report.update(pair_age_max_s=0.0, pair_skew_max_s=0.0)
            while time.monotonic() - started < duration:
                value = recovery.poll(cameras.snapshot)
                if value is None:
                    time.sleep(1 / cfg.control.hz)
                    continue
                _, captured, _, skew, _ = value
                age = time.monotonic() - captured
                report["samples"] += 1
                age_total += age
                report["pair_age_mean_s"] = age_total / report["samples"]
                report["pair_age_max_s"] = max(report["pair_age_max_s"], age)
                report["pair_skew_max_s"] = max(report["pair_skew_max_s"], skew)
                report["elapsed_s"] = time.monotonic() - started
                time.sleep(1 / cfg.control.hz)
            report["outcome"] = "camera_test_complete"
        except KeyboardInterrupt:
            report["outcome"] = "aborted"
        except Exception as exc:
            report.update(outcome="fault", error=f"{type(exc).__name__}: {exc}")
        finally:
            report["camera_recovery"] = recovery.summary()
            if cameras is not None:
                report["camera_diagnostics"] = cameras.diagnostics()
                try:
                    cameras.close()
                except Exception as exc:
                    report.update(outcome="fault", close_error=str(exc))
            report["finished_wall"] = time.time()
    return report
