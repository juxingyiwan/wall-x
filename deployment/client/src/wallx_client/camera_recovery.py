"""Bound transient camera gaps without relaxing image or motor freshness checks."""

import time

import numpy as np

from .contracts import CameraUnavailable, HardwareError


class CameraRecoveryBudget:
    """One bounded outage, shared by capture, pairing and observation readers.

    A gap gets one timeout. An actual device reopen started within that window
    gets one timeout from its start, including capture warmup and pairing. Other
    reopens in the same outage cannot extend it. Call under the camera lock.
    """

    def __init__(self, timeout_s):
        self.timeout_s = timeout_s
        self.clear()

    def clear(self):
        self.deadline = None
        self.reopened = False

    def check(self, now):
        if self.deadline is not None and now >= self.deadline:
            raise HardwareError("No fresh synchronized camera pair within the recovery timeout")

    def wait(self, now):
        self.check(now)
        if self.deadline is None:
            self.deadline = now + self.timeout_s
        return self.deadline

    def reopen(self, now):
        self.wait(now)  # An expired gap must never be revived by a late reconnect.
        if not self.reopened:
            self.deadline = now + self.timeout_s
            self.reopened = True
        return self.deadline


class CameraRecovery:
    """Nonblocking observation gate shared by both controllers and diagnostics.

    Only CameraUnavailable is recoverable. A caller must continue reading motor
    feedback and sending its hold/braking targets while poll() returns None.
    No stale image or fabricated Observation is returned to inference/recording.
    """

    def __init__(self, timeout_s, event):
        self.timeout_s, self.event = timeout_s, event
        self.started = None
        self.deadline = None
        self.capture_owns_deadline = False
        self.entered = self.recovered = False
        self.gaps = self.recoveries = self.wait_ticks = 0
        self.max_wait_s = 0.0
        self.camera_generation = None

    def poll(self, read):
        self.entered = self.recovered = False
        try:
            value = read()
            # A reconnect can finish between control ticks. Force one hold/brake
            # tick even if this caller never observed a missing frame, so queued
            # and in-flight actions from the preceding stream are invalidated.
            obs = value[0] if isinstance(value, tuple) and value else value
            generation = getattr(obs, "camera_generation", None)
            previous = self.camera_generation
            self.camera_generation = generation
            if previous is not None and generation != previous:
                self.event("camera_generation_changed", previous=previous, current=generation)
                if self.started is None:
                    raise CameraUnavailable("Camera stream reconnected; discard old predictions")
                # An observed gap already invalidated the old predictions. Do
                # not turn its in-time recovery into another missing-frame tick.
        except CameraUnavailable as exc:
            now = time.monotonic()
            if self.started is None:
                self.started = now
                self.deadline = now + self.timeout_s
                self.capture_owns_deadline = False
                self.entered = True
                self.gaps += 1
                self.event("camera_wait_started", reason=str(exc), timeout_s=self.timeout_s)
            if exc.recovery_deadline is not None:
                self.capture_owns_deadline = True
                # Use the capture owner's absolute deadline, not a new timeout
                # on each control tick. Even a faulty provider cannot extend
                # an outage beyond the gap + one reopen budget.
                deadline = min(exc.recovery_deadline, self.started + 2 * self.timeout_s)
                if deadline > self.deadline:
                    self.event("camera_wait_reopening", deadline_monotonic=deadline)
                self.deadline = deadline
            self.wait_ticks += 1
            self._check_deadline(now)
            return None
        if self.started is not None:
            # Capture enforces its deadline before publishing a recovered pair.
            # The next control tick may arrive just after that deadline: do not
            # reject a pair that the provider already accepted in time.
            elapsed = self._check_deadline(
                time.monotonic(), recovered_by_capture=self.capture_owns_deadline
            )
            self.recovered = True
            self.recoveries += 1
            self.started = None
            self.deadline = None
            self.event("camera_wait_recovered", elapsed_s=elapsed)
        return value

    def _check_deadline(self, now, *, recovered_by_capture=False):
        elapsed = now - self.started
        self.max_wait_s = max(self.max_wait_s, elapsed)
        deadline = self.started + 2 * self.timeout_s if recovered_by_capture else self.deadline
        if now >= deadline:
            raise HardwareError(f"Camera recovery timed out after {elapsed:.3f}s")
        return elapsed

    def summary(self):
        return dict(
            gaps=self.gaps,
            recoveries=self.recoveries,
            wait_ticks=self.wait_ticks,
            max_wait_s=self.max_wait_s,
        )


def record_camera_hold(hw, recorder, raw, state, status, *, generation):
    """Send a motor-only hold/brake tick without logging an expired image."""
    target = hw.send(raw)
    recorder.append(
        "command",
        phase="camera_wait",
        generation=generation,
        observed_physical=np.asarray(state).tolist(),
        executed_recorded_command=np.asarray(raw).tolist(),
        image_monotonic=None,
        sent_monotonic=time.monotonic(),
        **status,
        **target,
    )
    return target
