"""Shared episode limits and orderly shutdown for synchronous and RTC control.

An execution budget bounds an unattended retry loop; it does not infer task
success or semantic progress. Only the control thread owns motor commands.
"""

import signal
import threading
import time
from contextlib import contextmanager


class EpisodeLimit(Exception):
    """The configured execution budget has been consumed."""


class EpisodeBudget:
    def __init__(self, control, started):
        self.deadline = started + control.duration_s
        self.max_policy_chunks = control.max_policy_chunks
        self.chunks = 0

    def check_time(self, now):
        if now >= self.deadline:
            raise EpisodeLimit("duration_s")

    def require_plan_capacity(self):
        if self.chunks >= self.max_policy_chunks:
            raise EpisodeLimit("max_policy_chunks")

    def plan_started(self):
        self.require_plan_capacity()
        self.chunks += 1


@contextmanager
def protect_shutdown():
    """Defer ordinary termination signals during bounded motor cleanup.

    Repeated Ctrl+C must not interrupt homing or launch a second CAN owner. This
    does not intercept SIGKILL or replace a physical emergency stop.
    """
    previous = {}
    if threading.current_thread() is threading.main_thread():
        for name in ("SIGINT", "SIGTERM", "SIGHUP"):
            if sig := getattr(signal, name, None):
                previous[sig] = signal.signal(sig, lambda *_: None)
    try:
        yield
    finally:
        for sig, handler in previous.items():
            signal.signal(sig, handler)


def finish_episode(hw, connection, report, pending, *workers, timeout_s=2.0):
    """Invalidate pending work, close hardware, then join inference workers.

    Inference workers cannot send motor commands. Closing the transport
    or waiting for a worker must never delay the start of hardware cleanup.
    """
    report["stop_outcome"] = report["outcome"]
    errors = []
    from .workers import bounded_call

    if pending is not None:
        report["pending_cancelled"] = pending.cancel()
    with protect_shutdown():
        try:
            hw.close()
        except BaseException as exc:
            report.update(outcome="fault", shutdown_error=f"{type(exc).__name__}: {exc}")
        finally:
            if hasattr(hw, "shutdown_report"):
                report["shutdown"] = dict(hw.shutdown_report)
    # Signal handlers are restored before any potentially blocked background work.
    deadline = time.monotonic() + timeout_s
    report["transport_close_attempted"] = True
    report["worker_cleanup"] = []
    try:
        remaining = max(0, deadline - time.monotonic()) / 2
        if hasattr(connection, "close_with_status"):
            # Leave time for the backend to return its status after a timed join.
            status = bounded_call(lambda: connection.close_with_status(remaining * 0.95), remaining)
            report["backend_cleanup"] = status
            report["worker_cleanup"].append(status)
            if status["timed_out"]:
                errors.append(f"backend cleanup: {status['status']}")
        else:
            bounded_call(connection.close, remaining)
    except Exception as exc:
        report["backend_cleanup"] = {
            "kind": "connection",
            "status": "unconfirmed",
            "timed_out": True,
        }
        errors.append(f"transport close: {type(exc).__name__}: {exc}")
    for worker in reversed(workers):
        try:
            remaining = max(0, deadline - time.monotonic())
            if hasattr(worker, "close"):
                status = worker.close(remaining)
            else:
                bounded_call(
                    lambda owner=worker: owner.shutdown(wait=True, cancel_futures=True), remaining
                )
                status = {"kind": "external", "status": "joined", "timed_out": False}
            report["worker_cleanup"].append(status)
            if status["timed_out"]:
                errors.append(f"worker cleanup: {status['status']}")
        except Exception as exc:
            errors.append(f"worker shutdown: {type(exc).__name__}: {exc}")
    if errors:
        report.update(outcome="fault", cleanup_errors=errors)
    report["finished_wall"] = time.time()
