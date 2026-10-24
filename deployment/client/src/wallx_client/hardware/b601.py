"""B601 RS adapter using the installed LeRobot/MotorBridge driver.

The explicit observation-only path follows the existing Cosmos deployment:
opening the bus and requesting telemetry must not call Robot.connect/configure,
which can enable motors or calibrate on some driver versions.
"""

import hashlib
import inspect
import logging
import math
import os
import threading
import time
from collections import deque
from itertools import product
from pathlib import Path

import numpy as np

from ..camera_recovery import CameraRecoveryBudget
from ..config import ControlConfig, ShutdownConfig
from ..contracts import JOINTS, CameraUnavailable, HardwareError, Observation
from ..lifecycle import protect_shutdown
from .camera_metrics import CameraMetrics, CaptureClock
from .front_home import (
    CLEARANCE_M,
    FORWARD_LIMIT_M,
    MAX_ACCEL_DEG_S2,
    MAX_SPEED_DEG_S,
    execute_home,
    plan_home,
)
from .home_geometry import B601HomeGeometry
from .homing import homing_deadline
from .robstride_temperature import SignedTemperatureMotor, decode_temperature_word
from .temperature_guard import TemperatureFeedbackInvalid, TemperatureGuard
from .zero_servo import (
    MAX_REFINEMENT_S,
    ZERO_PRECISION_DEG,
    corrected_gripper_torque,
    zero_gripper_session,
)

# V4L2 uses whole seconds and caches this setting on its first select().
# Bound each select wait, not the duration of arbitrary kernel/codec calls.
V4L2_SELECT_TIMEOUT_S = 1


class StaleFeedback(HardwareError):
    """Stop policy execution; only shutdown may revalidate feedback for homing."""


class Feedback:
    def __init__(self, robot, cfg, event=None):
        import can

        self.robot, self.cfg = robot, cfg
        self.bus = can.Bus(
            channel=cfg.can_channel, interface="socketcan", receive_own_messages=False
        )
        self.latest = {}
        self.raw_frames = deque(maxlen=512)
        self.last_sdk_states = ()
        self.event = event
        self.temperature_guard = TemperatureGuard(
            cfg.max_temperature_c,
            cfg.temperature_feedback_grace_s,
            self._temperature_event,
            minimum=cfg.min_temperature_c,
        )
        self.fault_recorded = False
        self.error = None
        self.receive_error = None
        self.shutdown_only = False
        self.lock = threading.Lock()
        self.stop = threading.Event()
        self.thread = threading.Thread(target=self._receive, name="can-feedback", daemon=True)
        self.thread.start()

    def _receive(self):
        try:
            while not self.stop.is_set():
                msg = self.bus.recv(timeout=0.02)
                if msg is None:
                    continue
                self._remember_frame(msg)
                if msg.is_error_frame:
                    raise HardwareError("SocketCAN reported an error frame")
                if msg.is_rx and msg.is_extended_id and ((msg.arbitration_id >> 24) & 31) == 2:
                    motor_id = (msg.arbitration_id >> 8) & 255
                    fault_bits = (msg.arbitration_id >> 16) & 63
                    if msg.dlc != 8 or len(msg.data) != 8:
                        raise HardwareError(f"Motor {motor_id}: malformed status frame")
                    # A version reply shares type 2 but contains no measurements.
                    if bytes(msg.data[:3]) == b"\x00\xc4\x56":
                        motor_ids = {value[0] for value in self.robot.config.motor_can_ids.values()}
                        if fault_bits and motor_id in motor_ids:
                            raise HardwareError(
                                f"Motor {motor_id}: CAN status fault bits={fault_bits}"
                            )
                        continue
                    temperature_word = int.from_bytes(msg.data[6:8], "big")
                    with self.lock:
                        self.latest[motor_id] = (msg.timestamp, fault_bits, temperature_word)
                    motor_ids = {value[0] for value in self.robot.config.motor_can_ids.values()}
                    if fault_bits and motor_id in motor_ids:
                        raise HardwareError(f"Motor {motor_id}: CAN status fault bits={fault_bits}")
        except Exception as exc:
            # Separate the receiver's permanent fault from a control-thread
            # stale latch, so shutdown recovery cannot erase a CAN/read error.
            with self.lock:
                self.receive_error = exc

    def _remember_frame(self, msg):
        if not msg.is_rx:
            return
        with self.lock:
            self.raw_frames.append(
                dict(
                    timestamp=msg.timestamp,
                    arbitration_id=msg.arbitration_id,
                    is_extended_id=msg.is_extended_id,
                    is_error_frame=msg.is_error_frame,
                    dlc=msg.dlc,
                    data_hex=bytes(msg.data).hex(),
                )
            )

    def record_fault(self, reason, *, native_temperatures=None):
        """Persist evidence once; a logger failure must not interrupt motor cleanup.

        SDK snapshots and this independent SocketCAN listener are asynchronous.
        Keep frame timestamps so diagnostics do not assume they are the same frame.
        """
        if getattr(self, "event", None) is None or self.fault_recorded:
            return
        self.fault_recorded = True
        try:
            with self.lock:
                frames = list(self.raw_frames)

            def serializable(value):
                return (
                    str(value) if isinstance(value, float) and not math.isfinite(value) else value
                )

            fields = ("arbitration_id", "status_code", "pos", "vel", "torq", "t_mos", "t_rotor")
            states = {
                name: {key: serializable(getattr(state, key, None)) for key in fields}
                for name, state in zip(JOINTS, self.last_sdk_states, strict=False)
            }
            sdk_temperatures = {
                name: serializable(motor.last_sdk_state.t_mos)
                for name, motor in getattr(getattr(self, "robot", None), "motors", {}).items()
                if isinstance(motor, SignedTemperatureMotor) and motor.last_sdk_state is not None
            }
            self.event(
                "motor_feedback_fault",
                reason=reason,
                temperature_limit_c=self.cfg.max_temperature_c,
                temperature_min_c=self.cfg.min_temperature_c,
                last_snapshot_states=states,
                temperature_decoding="legacy big-endian int16 / 10 interpretation",
                temperature_signedness_verified=False,
                fault_type=type(
                    getattr(self, "error", None) or self.temperature_guard.error
                ).__name__,
                sdk_unsigned_temperatures_c=sdk_temperatures,
                native_temperatures_c={
                    name: serializable(value) for name, value in (native_temperatures or {}).items()
                },
                recent_can_frames=frames,
            )
        except Exception:
            pass

    def _temperature_event(self, kind, **fields):
        if kind == "temperature_feedback_transient":
            logging.getLogger(__name__).warning(
                "%s: transient temperature feedback %.1f°C; grace %.2fs",
                fields["joint"],
                fields["decoded_c"],
                fields["grace_s"],
            )
        else:
            logging.getLogger(__name__).info("%s: temperature feedback recovered", fields["joint"])
        if self.event is not None:
            with self.lock:
                frames = list(self.raw_frames)
            self.event(kind, **fields, recent_can_frames=frames)

    def check_temperatures(self, temperatures):
        check = (
            self.temperature_guard.check_for_shutdown
            if getattr(self, "shutdown_only", False)
            else self.temperature_guard.check
        )
        try:
            return check(temperatures, time.monotonic())
        except HardwareError as exc:
            self.error = exc
            raise

    def begin_shutdown(self):
        """Irreversible switch to measured-only temperature checks for cleanup."""
        self.shutdown_only = True

    def snapshot(self):
        try:
            return self._snapshot()
        except Exception as exc:
            # Policy execution remains stopped even if fresh frames arrive.
            self.error = exc
            self.record_fault(str(exc))
            raise

    def _check_error(self, *, allow_recovery=False):
        if getattr(self, "receive_error", None) is not None:
            raise HardwareError(f"CAN receiver failed: {self.receive_error}")
        if self.error is not None and not (
            allow_recovery and isinstance(self.error, (StaleFeedback, TemperatureFeedbackInvalid))
        ):
            raise self.error

    def _snapshot(self, *, allow_recovery=False):
        self._check_error(allow_recovery=allow_recovery)
        with self.lock:
            latest = dict(self.latest)
        ids = [self.robot.config.motor_can_ids[name][0] for name in JOINTS]
        now = time.time()
        ages = [now - latest.get(i, (-1e9, 0))[0] for i in ids]
        if not all(math.isfinite(age) for age in ages) or min(ages) < -0.02:
            raise HardwareError("Invalid CAN feedback timestamp")
        if max(ages) > self.cfg.max_feedback_age_s:
            expired = {
                name: round(age, 3)
                for name, age in zip(JOINTS, ages, strict=True)
                if age > self.cfg.max_feedback_age_s or age < -0.02
            }
            raise StaleFeedback(f"Stale CAN feedback: max age={max(ages):.3f}s; joints={expired}")
        self.robot.bus.poll_feedback_once()
        states = [self.robot.motors[name].get_state() for name in JOINTS]
        self.last_sdk_states = tuple(states)
        for name, motor_id, state in zip(JOINTS, ids, states, strict=True):
            if state is None or latest[motor_id][1] or state.status_code != 0:
                raise HardwareError(f"{name}: missing or faulted motor feedback")
        position = np.asarray([math.degrees(state.pos) for state in states], dtype=np.float32)
        if not np.isfinite(position).all():
            raise HardwareError("Non-finite motor position")
        temperatures = {name: state.t_mos for name, state in zip(JOINTS, states, strict=True)}
        checked_temperatures = temperatures
        if getattr(self, "shutdown_only", False):
            # A healthy SDK cache cannot prove that new CAN samples recovered.
            # Both independent measurements must pass; they need not be from
            # the same instant. Check them together so any high value wins
            # over an invalid value on another joint/source.
            checked_temperatures = {
                **temperatures,
                **{
                    f"{name} (CAN)": decode_temperature_word(latest[motor_id][2])
                    for name, motor_id in zip(JOINTS, ids, strict=True)
                },
            }
        temperature_status = self.check_temperatures(checked_temperatures)
        return position, {
            "feedback_age_max_s": max(ages),
            **temperature_status,
            "motor_temperature_c": temperatures,
        }

    def refresh(self):
        self._check_error()
        started = time.time()
        for motor in self.robot.motors.values():
            motor.robstride_set_active_report(True)
        deadline = time.monotonic() + 0.2
        ids = [self.robot.config.motor_can_ids[name][0] for name in JOINTS]
        while time.monotonic() < deadline:
            self._check_error()
            with self.lock:
                ready = all(self.latest.get(i, (-1, 0))[0] >= started for i in ids)
            if ready:
                return self.snapshot()
            time.sleep(0.002)
        self.error = StaleFeedback("No fresh motor status responses while observing")
        raise self.error

    def recover_for_shutdown(self, timeout_s, settle_s, hold):
        """Revalidate stale/invalid-temperature feedback with new seven-motor replies.

        Never enable, clear motor faults, relax thresholds, or resume a policy.
        A hold is sent only after fresh, healthy feedback from every motor; it
        preserves the last target rather than removing its gravity-support offset.
        The caller owns a process deadline as well as this polling deadline.
        """
        self.begin_shutdown()
        deadline = time.monotonic() + timeout_s
        stable_since = None
        last_valid = None
        samples = 0
        ids = [self.robot.config.motor_can_ids[name][0] for name in JOINTS]
        try:
            while time.monotonic() < deadline:
                self._check_error(allow_recovery=True)
                requested = time.time()
                for motor in self.robot.motors.values():
                    motor.robstride_set_active_report(True)
                reply_deadline = min(deadline, time.monotonic() + self.cfg.max_feedback_age_s)
                ready = False
                while time.monotonic() < reply_deadline:
                    self._check_error(allow_recovery=True)
                    with self.lock:
                        ready = all(self.latest.get(i, (-1, 0))[0] >= requested for i in ids)
                    if ready:
                        break
                    time.sleep(0.002)
                if ready:
                    try:
                        measured, status = self._snapshot(allow_recovery=True)
                    except (StaleFeedback, TemperatureFeedbackInvalid):
                        ready = False
                    else:
                        # Tolerated below-minimum temperatures are not enough to
                        # declare recovery and start a new homing trajectory.
                        ready = not status.get("temperature_feedback_pending")
                if ready:
                    now = time.monotonic()
                    if now >= deadline:
                        break
                    if last_valid is not None and now - last_valid > self.cfg.max_feedback_age_s:
                        stable_since = None
                    last_valid = now
                    if stable_since is None:
                        stable_since, samples = now, 0
                    samples += 1
                    hold(measured)
                    self._check_error(allow_recovery=True)
                    if time.monotonic() >= deadline:
                        break
                    if now - stable_since >= settle_s and samples >= 2:
                        with self.lock:
                            self._check_error(allow_recovery=True)
                            self.error = None
                        return measured, status
                else:
                    stable_since, samples = None, 0
                time.sleep(min(1 / 30, max(0, deadline - time.monotonic())))
            raise HardwareError(f"Shutdown feedback did not recover within {timeout_s:g}s")
        except Exception as exc:
            self.error = exc
            self.record_fault(str(exc))
            raise

    def close(self):
        self.stop.set()
        self.thread.join(timeout=1)
        self.bus.shutdown()


class Cameras:
    """Independent USB readers with bounded, timestamp-matched frame buffers."""

    def __init__(self, cfg, event=None):
        self.cfg = cfg
        self.event = event
        self.devices = {}
        self.latest = None
        self.error = None
        self.stop = threading.Event()
        self.lock = threading.Lock()
        self.buffers = {}
        self.workers = []
        self.metrics = {name: CameraMetrics() for name in cfg.camera_devices}
        self.formats = {}
        self.generation = 0
        self.latest_generation = 0
        self.reopen_counts = dict.fromkeys(cfg.camera_devices, 0)
        self.reopening = {}
        self.recovery = CameraRecoveryBudget(cfg.camera_recovery_timeout_s)
        self.consumed = dict.fromkeys(cfg.camera_devices, -1)
        try:
            for name in cfg.camera_devices:
                self.buffers[name] = deque(maxlen=6)
                self.devices[name], self.formats[name] = self._open_camera(name)
        except BaseException:
            for cap in self.devices.values():
                cap.release()
            raise
        for name, cap in list(self.devices.items()):
            worker = threading.Thread(
                target=self._read_camera, args=(name, cap), name=f"camera-{name}", daemon=True
            )
            self.workers.append(worker)
        self.thread = threading.Thread(target=self._capture, name="camera-pair", daemon=True)
        try:
            for worker in self.workers:
                worker.start()
            self.thread.start()
        except BaseException:
            self.stop.set()
            for worker in self.workers:
                if worker.ident is not None:
                    worker.join(timeout=2)
            # Unstarted workers never acquired ownership of their handles.
            for name, worker in zip(cfg.camera_devices, self.workers, strict=True):
                if worker.ident is None:
                    self.devices.pop(name).release()
            raise

    def _open_camera(self, name):
        # OpenCV defaults to ten seconds: longer than the recovery budget.
        # Set before opening either device (including the first ioctl/select).
        os.environ["OPENCV_VIDEOIO_V4L_SELECT_TIMEOUT"] = str(V4L2_SELECT_TIMEOUT_S)
        import cv2

        cfg = self.cfg
        path = cfg.camera_devices[name]
        cap = cv2.VideoCapture(path, cv2.CAP_V4L2)
        try:
            if not cap.isOpened():
                raise HardwareError(f"Cannot open camera {name}: {path}")
            cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, cfg.width)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, cfg.height)
            cap.set(cv2.CAP_PROP_FPS, cfg.fps)
            if not cap.set(cv2.CAP_PROP_BUFFERSIZE, cfg.camera_buffer_size):
                raise HardwareError(f"Cannot set camera buffer count: {name}")
            fourcc = int(cap.get(cv2.CAP_PROP_FOURCC))
            fmt = dict(
                device=path,
                width=cap.get(cv2.CAP_PROP_FRAME_WIDTH),
                height=cap.get(cv2.CAP_PROP_FRAME_HEIGHT),
                reported_fps=cap.get(cv2.CAP_PROP_FPS),
                fourcc="".join(chr((fourcc >> (8 * i)) & 255) for i in range(4)),
                buffer_size=cap.get(cv2.CAP_PROP_BUFFERSIZE),
                select_timeout_s=V4L2_SELECT_TIMEOUT_S,
            )
            if fmt["buffer_size"] != cfg.camera_buffer_size:
                raise HardwareError(f"Camera buffer count readback mismatch: {name}")
            if (fmt["width"], fmt["height"], fmt["fourcc"]) != (cfg.width, cfg.height, "MJPG"):
                raise HardwareError(f"Camera format readback mismatch: {name}: {fmt}")
            if not math.isclose(fmt["reported_fps"], cfg.fps, rel_tol=0.01):
                raise HardwareError(f"Camera FPS readback mismatch: {name}: {fmt}")
            return cap, fmt
        except BaseException:
            cap.release()
            raise

    def _reopen_camera(self, name, cap, reason):
        """Called only by the capture owner; never release a blocked reader elsewhere."""
        with self.lock:
            if self.reopen_counts[name] >= self.cfg.camera_max_reopens:
                raise HardwareError(f"Camera reopen limit reached: {name}: {reason}")
            deadline = self.recovery.reopen(time.monotonic())
            self.reopen_counts[name] += 1
            self.generation += 1
            self.latest = None
            for frames in self.buffers.values():
                frames.clear()
            self.reopening[name] = deadline
            self.devices.pop(name, None)
        self._report("camera_reopen_started")
        cap.release()
        last_error = reason
        while not self.stop.is_set() and time.monotonic() < deadline:
            try:
                new_cap, fmt = self._open_camera(name)
            except Exception as exc:
                last_error = str(exc)
                self.stop.wait(min(0.05, max(0, deadline - time.monotonic())))
                continue
            if self.stop.is_set() or time.monotonic() >= deadline:
                new_cap.release()
                break
            with self.lock:
                self.devices[name] = new_cap
                self.formats[name] = fmt
            self._report("camera_handle_reopened")
            # Keep reopening latched until a fresh frame arrives by the deadline.
            return new_cap
        if self.stop.is_set():
            return None
        raise HardwareError(f"Camera reopen timed out: {name}: {last_error}")

    def _read_camera(self, name, cap):
        import cv2

        sequence = 0
        failures = 0
        capture_clock = CaptureClock(self.cfg.camera_buffer_size)
        try:
            while not self.stop.is_set():
                started = time.monotonic()
                captured = capture_clock.start(started)
                with self.lock:
                    frame_generation = self.generation
                    self.metrics[name].reading_since = started
                try:
                    if not cap.grab():
                        raise HardwareError(f"Camera grab failed: {name}")
                    grabbed = time.monotonic()
                    capture_clock.completed(grabbed)
                    ok, frame = cap.retrieve()
                    if (
                        not ok
                        or frame is None
                        or frame.shape != (self.cfg.height, self.cfg.width, 3)
                    ):
                        raise HardwareError(f"Invalid camera frame: {name}")
                except (HardwareError, cv2.error) as exc:
                    failures += 1
                    elapsed = time.monotonic() - started
                    timed_out = elapsed >= V4L2_SELECT_TIMEOUT_S
                    if timed_out:
                        self._report("camera_read_timeout")
                    with self.lock:
                        self.metrics[name].reading_since = None
                    # Waiting for three one-second failures can exhaust the
                    # outer budget before the owner gets a chance to reopen.
                    # Keep the existing tolerance for quick isolated failures.
                    if timed_out or failures >= self.cfg.camera_max_consecutive_failures:
                        if self.cfg.camera_max_reopens == 0:
                            raise HardwareError(
                                f"{name}: {failures} consecutive capture failures: {exc}"
                            ) from exc
                        # USB can disconnect again before a reopened handle's
                        # first frame. Retry within the same recovery deadline
                        # and reopen count; _reopen_camera enforces both bounds.
                        cap = self._reopen_camera(name, cap, f"{exc} (read elapsed {elapsed:.3f}s)")
                        if cap is None:
                            break
                        failures = 0
                        capture_clock = CaptureClock(self.cfg.camera_buffer_size)
                        continue
                    # A single bad dequeue is not a permanent worker failure.
                    # Pair freshness still gates every observation independently.
                    self.stop.wait(1 / self.cfg.fps)
                    continue
                failures = 0
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                decoded = time.monotonic()
                with self.lock:
                    if frame_generation != self.generation:
                        continue  # Discard frames read concurrently with another camera's reset.
                    if name in self.reopening:
                        if decoded >= self.reopening[name]:
                            raise HardwareError(
                                f"Camera first fresh frame missed reopen deadline: {name}"
                            )
                        if decoded - captured <= self.cfg.max_camera_age_s:
                            del self.reopening[name]
                    self.metrics[name].completed(started, grabbed, decoded)
                    self.buffers[name].append((sequence, captured, grabbed, time.time(), rgb))
                sequence += 1
                self.stop.wait(max(0, 1 / self.cfg.fps - (time.monotonic() - started)))
        except Exception as exc:
            with self.lock:
                self.metrics[name].error = f"{type(exc).__name__}: {exc}"
                self.error = exc
        finally:
            with self.lock:
                owned = self.devices.pop(name, None)
            if owned is not None:
                owned.release()

    def diagnostics(self):
        """Snapshot reader health and why the next pair cannot be selected.

        Capture-start timestamps stay conservative. Diagnostics never make an
        expired frame appear fresh or weaken the age/skew constraints.
        """
        now = time.monotonic()
        with self.lock:
            latest = self.latest
            cameras = {}
            fresh = []
            reasons = []
            for name, frames in self.buffers.items():
                pending = [f for f in frames if f[0] > self.consumed[name]]
                valid = [f for f in pending if now - f[1] <= self.cfg.max_camera_age_s]
                cameras[name] = {
                    **self.metrics[name].summary(now),
                    "format": self.formats[name],
                    "unconsumed_frames": len(pending),
                    "latest_frame_age_s": now - frames[-1][1] if frames else None,
                }
                if not pending:
                    reasons.append(f"no_new_frame:{name}")
                elif not valid:
                    reasons.append(f"expired_frames:{name}")
                fresh.append(valid)
            skews = [max(f[2] for f in pair) - min(f[2] for f in pair) for pair in product(*fresh)]
            min_skew = min(skews) if skews else None
            if min_skew is not None and min_skew > self.cfg.max_camera_skew_s:
                reasons.append("pair_skew_exceeded")
            return {
                "generation": self.generation,
                "reopen_counts": dict(self.reopen_counts),
                "reopening": list(self.reopening),
                "recovery_deadline_monotonic": self.recovery.deadline,
                "cameras": cameras,
                "pairs": 0 if latest is None else latest[4] + 1,
                "pair_age_s": None if latest is None else now - latest[1],
                "pair_skew_s": None if latest is None else latest[3],
                "next_pair_min_skew_s": min_skew,
                "next_pair_blockers": reasons,
                "max_camera_age_s": self.cfg.max_camera_age_s,
                "max_camera_skew_s": self.cfg.max_camera_skew_s,
                "error": str(self.error) if self.error else None,
            }

    def _report(self, kind):
        if getattr(self, "event", None) is not None:
            try:
                self.event(kind, **self.diagnostics())
            except Exception:
                pass  # The control thread still detects logger and camera failures.

    def _select_pair(self, buffers, consumed, now):
        candidates = [
            [frame for frame in frames if frame[0] > consumed[name]]
            for name, frames in buffers.items()
        ]
        valid = []
        for pair in product(*candidates):
            captured = min(frame[1] for frame in pair)
            grabbed = [frame[2] for frame in pair]
            skew = max(grabbed) - min(grabbed)
            if now - captured <= self.cfg.max_camera_age_s and skew <= self.cfg.max_camera_skew_s:
                valid.append((captured, -skew, pair))
        # Prefer the freshest pair, with the smallest skew as a tie breaker.
        return max(valid, key=lambda item: item[:2])[2] if valid else None

    def _capture(self):
        sequence = 0
        warmup_deadline = time.monotonic() + 3
        next_report = time.monotonic()
        consumed = dict.fromkeys(self.buffers, -1)
        waiting_since = None
        try:
            while not self.stop.is_set():
                if self.error:
                    self._report("camera_fault")
                    return
                now = time.monotonic()
                with self.lock:
                    self.recovery.check(now)
                    buffers = {name: list(frames) for name, frames in self.buffers.items()}
                    latest = self.latest
                    generation = self.generation
                    reopen_deadlines = list(self.reopening.values())
                if reopen_deadlines and now >= min(reopen_deadlines):
                    raise HardwareError("Camera reopen deadline exceeded")
                pair = self._select_pair(buffers, consumed, now)
                if pair is not None:
                    images = {name: frame[4] for name, frame in zip(buffers, pair, strict=True)}
                    captured = min(frame[1] for frame in pair)
                    skew = max(frame[2] for frame in pair) - min(frame[2] for frame in pair)
                    with self.lock:
                        if generation != self.generation or self.reopening:
                            continue
                        self.recovery.check(time.monotonic())
                        self.latest = (images, captured, min(f[3] for f in pair), skew, sequence)
                        self.latest_generation = generation
                        self.consumed = {
                            name: frame[0] for name, frame in zip(buffers, pair, strict=True)
                        }
                        self.recovery.clear()
                    consumed = {name: frame[0] for name, frame in zip(buffers, pair, strict=True)}
                    sequence += 1
                    if waiting_since is not None:
                        self._report("camera_pair_recovered")
                        waiting_since = None
                elif (latest is None and now >= warmup_deadline) or (
                    latest is not None and now - latest[1] > self.cfg.max_camera_age_s
                ):
                    if waiting_since is None:
                        waiting_since = now
                        self._report("camera_pair_waiting")
                    with self.lock:
                        self.recovery.wait(now)
                if now >= next_report:
                    self._report("camera_diagnostics")
                    next_report = now + 1.0
                self.stop.wait(0.002)
        except Exception as exc:
            self.error = exc
            self._report("camera_fault")

    def snapshot(self, *, include_generation=False):
        try:
            return self._snapshot(include_generation=include_generation)
        except CameraUnavailable:
            raise  # Recoverable gap; transition events are emitted once, not every tick.
        except HardwareError:
            self._report("camera_snapshot_fault")
            raise

    def _snapshot(self, *, include_generation=False):
        if self.error:
            raise HardwareError(f"Camera worker failed: {self.error}")
        with self.lock:
            now = time.monotonic()
            self.recovery.check(now)
            if self.latest is None:
                raise CameraUnavailable(
                    "Waiting for first camera pair", recovery_deadline=self.recovery.wait(now)
                )
            images, captured, wall, skew, sequence = self.latest
            if now - captured > self.cfg.max_camera_age_s:
                raise CameraUnavailable(
                    "Camera pair is stale", recovery_deadline=self.recovery.wait(now)
                )
            result = (
                {name: image.copy() for name, image in images.items()},
                captured,
                wall,
                skew,
                sequence,
            )
            return (*result, self.latest_generation) if include_generation else result

    def close(self):
        self.stop.set()
        deadline = time.monotonic() + 2
        for thread in [self.thread, *self.workers]:
            thread.join(timeout=max(0, deadline - time.monotonic()))
        if any(thread.is_alive() for thread in self.workers):
            # Releasing a VideoCapture concurrently with a blocked read can
            # crash the process. Preserve ownership until the reader exits.
            raise HardwareError("Camera capture did not stop within two seconds")
        # Each worker releases its own handle in finally, including after stop.


class B601:
    def __init__(self, cfg, event, *, shutdown=None):
        self.cfg, self.event = cfg, event
        self.shutdown_cfg = shutdown or ShutdownConfig()
        self.control_cfg = ControlConfig()
        self.robot = self.feedback = self.cameras = None
        self.enabled = []
        self.all_enabled = False
        self.last_target = None
        self._home_active = False
        self._home_progress_state = None
        self._last_home_feedback = None
        self._last_zero_refinement = None
        self._home_inhibited = None
        self._closed = False
        self.shutdown_report = {"homing": "not_required"}

    def inhibit_home(self, reason):
        """Latch an unsafe motor condition; only a new session can clear it."""
        if self._home_inhibited is None:
            self._home_inhibited = reason

    def _home_feedback(self, *, refresh=False):
        if self._home_inhibited:
            raise HardwareError(self._home_inhibited)
        if getattr(self.robot, "_emergency_disable_requested", False):
            raise HardwareError("Driver emergency disable is active")
        if self.feedback is None:
            raise HardwareError("No feedback monitor available for homing")
        try:
            result = self.feedback.refresh() if refresh else self.feedback.snapshot()
            self._record_home_feedback(result)
            return result
        except (StaleFeedback, TemperatureFeedbackInvalid):
            # Keep the typed latch for one shutdown-only recovery attempt.
            raise
        except Exception as exc:
            self.inhibit_home(str(exc))
            raise

    def open_observation(self):
        from lerobot_robot_seeed_b601 import SeeedB601RSFollower, SeeedB601RSFollowerConfig
        from motorbridge import Controller

        config = SeeedB601RSFollowerConfig(
            port=self.cfg.can_channel, can_adapter="socketcan", id=self.cfg.robot_id, cameras={}
        )
        if config.max_relative_target is not None:
            raise HardwareError("Expected the existing driver's max_relative_target=None")
        adapter = self

        class GuardedFollower(SeeedB601RSFollower):
            def mit_output_torque_limit(self, motor, pos_target):
                native = super().mit_output_torque_limit(motor, pos_target)
                return corrected_gripper_torque(self, motor, pos_target, native)

            def _read_motor_temperatures(self):
                temperatures = super()._read_motor_temperatures()
                try:
                    adapter.feedback.check_temperatures(temperatures)
                except TemperatureFeedbackInvalid as exc:
                    # This is not proof of overheat. Stop control, but retain
                    # the shutdown-only option to verify recovery before home.
                    adapter.feedback.error = exc
                    adapter.feedback.record_fault(str(exc), native_temperatures=temperatures)
                    raise
                except HardwareError as exc:
                    # The pinned driver otherwise raises KeyboardInterrupt for
                    # overheat. That must be a latched motor fault, not an
                    # operator interruption that permits automatic homing.
                    adapter.inhibit_home(str(exc))
                    adapter.feedback.record_fault(str(exc), native_temperatures=temperatures)
                    raise
                return temperatures

            def send_action(self, action):
                # The pinned driver's safe_zero dispatches every interpolated
                # frame through this public method. Preserve its native path,
                # but require a fresh response from every motor before each
                # frame. Native stage 1 omits the gripper, so its command-
                # driven feedback would otherwise expire despite a healthy bus.
                if adapter._home_active:
                    adapter._home_feedback(refresh=True)
                return super().send_action(action)

        self.robot = GuardedFollower(config)
        self.robot.bus = Controller(channel=self.cfg.can_channel)
        self.robot._add_motors_to_bus()
        self.robot.motors = {
            name: SignedTemperatureMotor(motor) for name, motor in self.robot.motors.items()
        }
        self.directions = np.asarray([config.joint_directions[j] for j in JOINTS], dtype=float)
        self.limits = np.asarray([config.joint_limits[j] for j in JOINTS], dtype=float)
        if not np.isfinite(self.directions).all() or np.any(self.directions == 0):
            raise HardwareError("Invalid driver direction/scale mapping")
        self.feedback = Feedback(self.robot, self.cfg, self._event)
        self.feedback.refresh()
        self.cameras = Cameras(self.cfg, event=self._event)
        deadline = time.monotonic() + 3
        while (
            self.cameras.latest is None
            and self.cameras.error is None
            and time.monotonic() < deadline
        ):
            time.sleep(0.02)
        self.cameras.snapshot()

    def manifest(self):
        module = Path(inspect.getfile(self.robot.safe_zero))
        return {
            "driver_source": str(module),
            "driver_source_sha256": hashlib.sha256(module.read_bytes()).hexdigest(),
            "temperature_decoding": "legacy big-endian int16 / 10 at MotorBridge get_state",
            "temperature_signedness_verified": False,
            "home_forward_limit_m": FORWARD_LIMIT_M,
            "home_forward_clearance_m": CLEARANCE_M,
            "home_max_speed_deg_s": MAX_SPEED_DEG_S,
            "home_max_accel_deg_s2": MAX_ACCEL_DEG_S2,
            "home_forward_reference": "base_link +X / gripper_end (nominal URDF)",
            "joint_order": list(JOINTS),
            "joint_directions": self.directions.tolist(),
            "joint_limits": self.limits.tolist(),
            "can_channel": self.cfg.can_channel,
            "camera_devices": self.cfg.camera_devices,
        }

    def observe(self, disabled=False):
        images, captured, wall, skew, sequence, generation = self.cameras.snapshot(
            include_generation=True
        )
        state, status = self.observe_state(disabled=disabled)
        obs = Observation(
            state, images, captured, wall, time.monotonic(), skew, sequence, generation
        )
        return obs, status

    def observe_state(self, disabled=False):
        """Keep motor safety checks active even when no camera pair is available."""
        return self.feedback.refresh() if disabled else self.feedback.snapshot()

    def send(self, raw):
        if self._closed:
            raise HardwareError("Cannot send commands after shutdown")
        return self._send(raw)

    def _send(self, raw):
        """Dispatch a validated target; cleanup retains exclusive motor ownership."""
        raw = np.asarray(raw, dtype=float)
        if raw.shape != (7,) or not np.isfinite(raw).all():
            raise HardwareError("Expected seven finite recorded commands")
        physical = raw * self.directions
        expected = np.clip(physical, *self.limits.T)
        try:
            result = self.robot.send_action(
                dict(zip((j + ".pos" for j in JOINTS), raw.tolist(), strict=True))
            )
        except TemperatureFeedbackInvalid:
            # GuardedFollower rejected before command dispatch. Do not turn
            # a recoverable measurement fault into a permanent command fault.
            raise
        except Exception as exc:
            self.inhibit_home(f"Motor command failed: {exc}")
            raise
        target = np.asarray([result[j + ".pos"] for j in JOINTS])
        if not np.allclose(target, expected, atol=1e-5, rtol=0):
            self.inhibit_home("Driver target mapping mismatch")
            raise HardwareError(
                "Driver returned a target inconsistent with its direction/limit mapping"
            )
        self.last_target = target.copy()
        return {
            "recorded_command": raw.tolist(),
            "driver_target_physical": target.tolist(),
            "native_limit_clipped": (np.abs(expected - physical) > 1e-5).tolist(),
        }

    def hold_command(self, measured):
        """Keep the last dispatched target, bootstrapping from feedback only once."""
        # A loaded MIT position controller needs a target/feedback offset to
        # support gravity. Rebasing a hold to feedback removes that position
        # torque and can make the arm sag, even though motors remain enabled.
        # Preserve all seven targets during pauses, camera gaps and retries.
        target = np.asarray(measured if self.last_target is None else self.last_target, dtype=float)
        return np.clip(target, *self.limits.T) / self.directions

    def enable(self, control):
        if self._closed:
            raise HardwareError("Cannot enable motors after shutdown")
        self.control_cfg = control
        measured, _ = self.feedback.refresh()
        raw_hold = self.hold_command(measured)
        self.send(raw_hold)
        for name in JOINTS:
            self.enabled.append(name)  # Include partial/uncertain enable in cleanup.
            self.robot.motors[name].enable()
            self.send(raw_hold)
        self.all_enabled = True
        self.event("motors_enabled")
        if control.startup_home:
            self._home(control)

    def _home(self, control, *, phase="startup"):
        """Use the existing calibrated zero for both startup and shutdown."""
        self._event(f"{phase}_home_started")
        self._last_home_feedback = None
        self._last_zero_refinement = None
        self._home_progress_state = dict(phase=phase, stage="native_arm", emitted=None)
        try:
            self._native_home()
            self._home_progress_state.update(stage="gripper_close", emitted=None)
            with homing_deadline(self.shutdown_cfg.home_timeout_s):
                measured, _ = self._home_feedback(refresh=True)
                start = float(np.clip(measured[6], *self.limits[6]))
                frames = max(1, math.ceil(abs(start) / control.startup_gripper_speed_deg_s * 30))
                for i in range(1, frames + 1):
                    self._home_feedback(refresh=True)
                    target = np.zeros(7)
                    target[6] = start * (1 - i / frames)
                    # Public send() stays closed to policy commands during cleanup.
                    self._send(target / self.directions)
                    time.sleep(1 / 30)
                self._home_progress_state.update(stage="verify_zero", emitted=None)
                measured = self._wait_home(control, phase=phase)
        except BaseException as exc:
            diagnostic = dict(
                phase=phase,
                stage=self._home_progress_state["stage"],
                error=f"{type(exc).__name__}: {exc}",
                last_feedback=self._last_home_feedback,
                refinement=self._last_zero_refinement,
            )
            # Preserve the last completed read. Do not perform CAN I/O after a timeout.
            exc.homing_context = diagnostic
            self._event("home_failed", **diagnostic)
            self.inhibit_home(
                f"{phase.capitalize()} home failed during {diagnostic['stage']}: {exc}"
            )
            raise
        finally:
            self._home_progress_state = None
        self._event(f"{phase}_home_complete", measured=measured)
        return measured

    def _record_home_feedback(self, feedback):
        """Record existing feedback without adding CAN reads or changing control."""
        progress = self._home_progress_state
        if progress is None:
            return
        try:
            now = time.monotonic()
            limits = [self.shutdown_cfg.arm_tolerance_deg] * 6 + [
                min(self.control_cfg.settle_gripper_deg, ZERO_PRECISION_DEG)
            ]
            values = np.asarray(feedback[0], dtype=float)
            self._last_home_feedback = dict(
                phase=progress["phase"],
                stage=progress["stage"],
                feedback_monotonic=now,
                measured_physical_deg=values.tolist(),
                arm_error_deg=float(np.max(np.abs(values[:6]))),
                gripper_error_deg=float(abs(values[6])),
                tolerance_deg=limits,
                gripper_servo=dict(
                    getattr(getattr(self.robot, "_homing_zero_servo", None), "last", {})
                ),
                outside_tolerance=[
                    name
                    for name, value, limit in zip(JOINTS, values, limits, strict=True)
                    if abs(value) > limit
                ],
            )
            if progress["emitted"] is None or now - progress["emitted"] >= 1.0:
                progress["emitted"] = now
                self._event("home_progress", **self._last_home_feedback)
                logging.getLogger(__name__).info(
                    "Homing %s/%s: arm=%.3fdeg, gripper=%.3fdeg, outside=%s",
                    progress["phase"],
                    progress["stage"],
                    self._last_home_feedback["arm_error_deg"],
                    self._last_home_feedback["gripper_error_deg"],
                    self._last_home_feedback["outside_tolerance"],
                )
        except Exception:
            pass  # Diagnostics must never interrupt homing or motor cleanup.

    def _wait_home(self, control, *, phase="startup"):
        """Close the residual with feedback correction scoped to homing only."""
        with zero_gripper_session(self.robot) as servo:
            return self._settle_home(control, servo, phase=phase)

    def _settle_home(self, control, servo, *, phase="startup"):
        """Confirm measured arrival, not just completion of the zero trajectory.

        Native safe_zero only guarantees the six-arm trajectory; its conditional
        170-degree gripper target is not a gripper zero. Both entry and exit must
        explicitly close the gripper and verify all seven measured positions.
        No motor zero register or LeRobot calibration file is changed.
        """
        deadline = time.monotonic() + min(self.shutdown_cfg.home_timeout_s, MAX_REFINEMENT_S)
        precision = min(control.settle_gripper_deg, ZERO_PRECISION_DEG)
        stable_since = None
        while True:
            measured, _ = self._home_feedback(refresh=True)
            now = time.monotonic()
            arm_error = float(np.max(np.abs(measured[:6])))
            gripper_error = float(abs(measured[6]))
            arrival = servo.observe(float(measured[6]), now, precision)
            self._last_zero_refinement = dict(
                precision_deg=precision, arrival=arrival, torque=dict(servo.last)
            )
            if (
                arm_error <= self.shutdown_cfg.arm_tolerance_deg
                and gripper_error <= precision
                and arrival["stationary"]
            ):
                if stable_since is None:
                    stable_since = now
                if now - stable_since >= self.shutdown_cfg.settle_s:
                    self._event(
                        "zero_refinement_complete",
                        phase=phase,
                        measured_physical_deg=measured.tolist(),
                        **self._last_zero_refinement,
                    )
                    return measured.tolist()
            else:
                stable_since = None
            if arrival["stalled"]:
                reason = (
                    f"{phase.capitalize()} gripper zero stalled at {measured[6]:.3f}deg; "
                    f"target=0deg, no measurable progress for {arrival['no_progress_s']:.2f}s"
                )
                self.inhibit_home(reason)
                raise HardwareError(reason)
            if now >= deadline:
                reason = (
                    f"{phase.capitalize()} zero not reached: arm={arm_error:.3f}deg, "
                    f"gripper={gripper_error:.3f}deg"
                )
                self.inhibit_home(reason)
                raise HardwareError(reason)
            self._send(np.zeros(7))
            time.sleep(1 / 30)

    def _native_home(self):
        """Check the full forward sweep, then preserve the native finish."""
        codec = getattr(self, "_home_geometry", None)
        if codec is None:
            self._home_geometry = codec = B601HomeGeometry()
        if not np.array_equal(self.directions, codec.scale):
            raise HardwareError(
                "Driver direction/scale differs from the audited B601 home geometry"
            )
        measured, _ = self._home_feedback(refresh=True)
        path = plan_home(codec, measured, self.limits)
        self._event(
            "home_forward_plan",
            phase=(self._home_progress_state or {}).get("phase"),
            **path.summary(),
        )
        if not path.native:
            with homing_deadline(self.shutdown_cfg.home_timeout_s):
                execute_home(self, codec, path)
            if self._home_progress_state is not None:
                self._home_progress_state.update(stage="native_arm", emitted=None)
        return self._native_arm_home()

    def _native_arm_home(self):
        """Run native two-stage arm homing, then verify fresh measured arrival.

        No shell reset process is started: this session retains the shared robot locks
        and CAN connection. _home subsequently closes and verifies the gripper.
        """
        completed = []
        errors = []
        event = self._event

        class Observer(logging.Handler):
            def emit(self, record):
                message = record.getMessage()
                event("driver_home", level=record.levelname, message=message)
                if message == "safe_zero done.":
                    completed.append(True)
                if record.levelno >= logging.ERROR:
                    errors.append(message)

        logger = logging.getLogger("lerobot_robot_seeed_b601.seeed_b601_follower")
        handler, old_level = Observer(), logger.level
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
        try:
            with homing_deadline(self.shutdown_cfg.home_timeout_s):
                self._home_feedback(refresh=True)
                self._home_active = True
                self.robot.safe_zero(exit_on_complete=False)
                if not completed or errors:
                    raise HardwareError(f"Native home did not complete: {errors}")
                stable_since = None
                while True:
                    measured, _ = self.feedback.refresh()
                    self._home_feedback()
                    now = time.monotonic()
                    if np.max(np.abs(measured[:6])) <= self.shutdown_cfg.arm_tolerance_deg:
                        if stable_since is None:
                            stable_since = now
                        if now - stable_since >= self.shutdown_cfg.settle_s:
                            return measured.tolist()
                    else:
                        stable_since = None
                    time.sleep(0.02)
        except BaseException as exc:
            self.inhibit_home(f"Native home failed: {exc}")
            raise
        finally:
            self._home_active = False
            logger.removeHandler(handler)
            logger.setLevel(old_level)

    def _event(self, kind, **values):
        try:
            self.event(kind, **values)
        except Exception:
            pass  # Disk/telemetry failure must not interrupt motor cleanup.

    def close(self):
        if self._closed:
            return self.shutdown_report
        self._closed = True
        with protect_shutdown():
            return self._close()

    def _close(self):
        errors = []
        self.shutdown_report["started_wall"] = time.time()
        self.shutdown_report["feedback_recovery"] = "not_required"
        # Prefer verified home, including after a recoverable feedback gap.
        # Never re-enable motors or clear a hard fault just to make home possible.
        if self.robot and self.all_enabled:
            try:
                self._prepare_shutdown_home()
            except BaseException as exc:
                self.shutdown_report.update(homing="skipped", homing_error=str(exc))
                errors.append(f"Homing inhibited: {exc}")
            else:
                self.shutdown_report["homing"] = "running"
                try:
                    measured = self._home(self.control_cfg, phase="shutdown")
                    self.shutdown_report.update(homing="completed", measured_physical_deg=measured)
                except BaseException as exc:
                    self.shutdown_report.update(homing="failed", homing_error=str(exc))
                    errors.append(f"Native shutdown: {type(exc).__name__}: {exc}")
        self.shutdown_report["disable_reason"] = (
            "home_verified"
            if self.shutdown_report["homing"] == "completed"
            else "home_failed"
            if self.shutdown_report["homing"] == "failed"
            else "home_unavailable"
            if self.all_enabled
            else "partial_enable"
            if self.enabled
            else "not_enabled"
        )
        # Do not call driver.disconnect(): it initiates a second, unguarded home
        # and may swallow failures. Explicitly disable every attempted motor.
        disable_errors = []
        if self.robot and self.robot.bus is not None:
            for name in reversed(self.enabled):
                try:
                    self.robot.motors[name].disable()
                except Exception as exc:
                    disable_errors.append(f"disable {name}: {exc}")
        elif self.enabled:
            disable_errors.append("CAN bus unavailable; motor disable could not be confirmed")
        errors.extend(disable_errors)
        self.shutdown_report["disable"] = (
            "failed" if disable_errors else ("commands_sent" if self.enabled else "not_required")
        )
        for component in (self.cameras, self.feedback):
            if component is not None:
                try:
                    component.close()
                except Exception as exc:
                    errors.append(str(exc))
        self.cameras = self.feedback = None
        if self.robot and self.robot.bus is not None:
            bus = self.robot.bus
            for close in [bus.close_bus, *(m.close for m in self.robot.motors.values()), bus.close]:
                try:
                    close()
                except Exception as exc:
                    errors.append(f"close bus/motor: {exc}")
            self.robot.bus = None
        self.enabled = []
        self.all_enabled = False
        self.shutdown_report.update(errors=errors, finished_wall=time.time())
        self._event("shutdown_complete", **self.shutdown_report)
        if errors:
            raise HardwareError("; ".join(errors))
        return self.shutdown_report

    def _prepare_shutdown_home(self):
        """One bounded recovery before home; a failed home is never restarted."""
        try:
            _, status = self._home_feedback()
            if status.get("temperature_feedback_pending"):
                raise TemperatureFeedbackInvalid("Temperature feedback still pending at shutdown")
            return
        except (StaleFeedback, TemperatureFeedbackInvalid) as exc:
            self.shutdown_report.update(
                feedback_recovery="running",
                feedback_fault=str(exc),
                feedback_fault_type=type(exc).__name__,
            )
            self._event(
                "shutdown_feedback_recovery_started", reason=str(exc), fault_type=type(exc).__name__
            )
        started = time.monotonic()

        def hold(measured):
            if self._home_inhibited or getattr(self.robot, "_emergency_disable_requested", False):
                raise HardwareError(self._home_inhibited or "Driver emergency disable is active")
            self._send(self.hold_command(measured))

        try:
            with homing_deadline(self.shutdown_cfg.feedback_recovery_timeout_s):
                self.feedback.recover_for_shutdown(
                    self.shutdown_cfg.feedback_recovery_timeout_s,
                    self.shutdown_cfg.settle_s,
                    hold,
                )
            self._home_feedback()
        except BaseException as exc:
            self.inhibit_home(f"Shutdown feedback recovery failed: {exc}")
            self.shutdown_report.update(
                feedback_recovery="failed", feedback_recovery_error=str(exc)
            )
            self._event("shutdown_feedback_recovery_failed", reason=str(exc))
            raise
        finally:
            self.shutdown_report["feedback_recovery_elapsed_s"] = time.monotonic() - started
        self.shutdown_report["feedback_recovery"] = "recovered"
        self._event("shutdown_feedback_recovered", elapsed_s=time.monotonic() - started)
