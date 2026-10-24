"""Replay negative feedback without opening CAN or extending the fault deadline."""

import threading
from collections import deque
from dataclasses import dataclass, replace
from types import SimpleNamespace

import pytest

from wallx_client.config import validate_config
from wallx_client.contracts import JOINTS, HardwareError
from wallx_client.hardware.b601 import Feedback
from wallx_client.hardware.robstride_temperature import SignedTemperatureMotor
from wallx_client.hardware.temperature_guard import TemperatureGuard


def read(guard, value, now):
    return guard.check({"shoulder_lift": value}, now)


@pytest.mark.parametrize("value", [-25.0, -20.0, -3.0, 0.0, 35.0, 60.0])
def test_configured_minus25_bound_applies_to_policy_and_shutdown(cfg, value):
    assert cfg.robot.min_temperature_c == -25.0
    guard = TemperatureGuard(65, 0.5, minimum=cfg.robot.min_temperature_c)
    status = read(guard, value, 0)
    assert status["temperature_max_c"] == value
    assert status["temperature_feedback_pending"] == {}
    assert guard.check_for_shutdown({"shoulder_lift": value}, 0.1)["temperature_max_c"] == value


@pytest.mark.parametrize("value", [-25.1, -30.0])
def test_below_minus25_still_requires_feedback_recovery(value):
    from wallx_client.hardware.temperature_guard import TemperatureFeedbackInvalid

    guard = TemperatureGuard(65, 0.5, minimum=-25)
    read(guard, 35, 0)
    assert read(guard, value, 0.1)["temperature_feedback_pending"]
    with pytest.raises(TemperatureFeedbackInvalid):
        guard.check_for_shutdown({"shoulder_lift": value}, 0.2)
    with pytest.raises(TemperatureFeedbackInvalid):
        read(guard, value, 0.5)


@pytest.mark.parametrize("value", [65.0, 100.0, float("nan"), float("inf")])
def test_minus25_lower_bound_does_not_relax_upper_or_nonfinite_checks(value):
    guard = TemperatureGuard(65, 0.5, minimum=-25)
    with pytest.raises(HardwareError):
        read(guard, value, 0)
    with pytest.raises(HardwareError):
        guard.check_for_shutdown({"shoulder_lift": value}, 0.1)


@pytest.mark.parametrize("value", [True, "-25", float("nan"), float("inf"), 60, 65])
def test_invalid_minimum_temperature_config_is_rejected(cfg, value):
    with pytest.raises(ValueError, match="min_temperature_c"):
        validate_config(replace(cfg, robot=replace(cfg.robot, min_temperature_c=value)))


def test_recorded_raw_sequence_can_recover_without_latching():
    from wallx_client.hardware.robstride_temperature import decode_temperature_word

    events = []
    guard = TemperatureGuard(65, 0.5, lambda kind, **kw: events.append((kind, kw)))
    # Received CAN sequence from the failing run, relative to its last 41 C frame.
    for stamp, word in [
        (0, 0x019A),
        (0.00998, 0x017C),
        (0.01998, 0x0014),
        (0.021991, 0xFFE2),
        (0.03, 0xFF38),
        (0.04, 0xFF38),
        (0.05, 0xFF38),
    ]:
        status = read(guard, decode_temperature_word(word), stamp)
    assert status["temperature_feedback_pending"]
    assert guard.error is None
    # Recovery is synthetic: the recorded run stopped before recovery was observed.
    read(guard, 41, 0.08)
    assert read(guard, 41, 0.179)["temperature_feedback_pending"]
    assert read(guard, 41, 0.181)["temperature_feedback_pending"] == {}
    assert [kind for kind, _ in events] == [
        "temperature_feedback_transient",
        "temperature_feedback_recovered",
    ]


@pytest.mark.parametrize("value", [-3, -20])
def test_persistent_negative_expires_and_latches_even_after_recovery(value):
    guard = TemperatureGuard(65, 0.5)
    read(guard, 41, 0)
    for now in [0.02, 0.1, 0.2, 0.3, 0.499]:
        assert read(guard, value, now)["temperature_max_c"] == 41
    with pytest.raises(HardwareError, match="within 0.5s"):
        read(guard, value, 0.5)
    with pytest.raises(HardwareError, match="within 0.5s"):
        read(guard, 41, 0.6)


def test_alternating_good_bad_does_not_reset_deadline():
    guard = TemperatureGuard(65, 0.5)
    read(guard, 41, 0)
    for now, value in [
        (0.03, -3),
        (0.1, 41),
        (0.16, -20),
        (0.2, 41),
        (0.26, -3),
        (0.3, 41),
        (0.36, -3),
        (0.45, 41),
    ]:
        read(guard, value, now)
    with pytest.raises(HardwareError, match="within 0.5s"):
        read(guard, 41, 0.5)


@pytest.mark.parametrize("value", [65, 100, float("nan"), float("inf"), -float("inf")])
def test_hard_temperature_fault_is_immediate_during_grace(value):
    guard = TemperatureGuard(65, 0.5)
    read(guard, 41, 0)
    read(guard, -3, 0.02)
    with pytest.raises(HardwareError):
        read(guard, value, 0.021)


@pytest.mark.parametrize("reference,now", [(None, 0), (61, 0.03), (41, 0.5)])
def test_missing_hot_or_expired_reference_is_not_tolerated(reference, now):
    guard = TemperatureGuard(65, 0.5)
    if reference is not None:
        read(guard, reference, 0)
    with pytest.raises(HardwareError, match="reference"):
        read(guard, -3, now)


def test_grace_can_be_disabled():
    guard = TemperatureGuard(65, 0)
    read(guard, 41, 0)
    with pytest.raises(HardwareError, match="disabled"):
        read(guard, -3, 0.001)


def test_recovery_near_limit_fails_immediately():
    guard = TemperatureGuard(65, 0.5)
    read(guard, 41, 0)
    read(guard, -3, 0.02)
    with pytest.raises(HardwareError, match="recovery too close"):
        read(guard, 61, 0.03)


def test_recovery_never_hides_a_higher_valid_temperature():
    guard = TemperatureGuard(65, 0.5)
    read(guard, 41, 0)
    read(guard, -3, 0.02)
    status = read(guard, 59, 0.03)
    assert status["temperature_max_c"] == 59
    assert status["temperature_feedback_pending"]


def test_repeated_episodes_have_a_per_joint_rolling_limit():
    guard = TemperatureGuard(65, 0.5)
    for start in [0, 1, 2]:
        read(guard, 41, start)
        read(guard, -3, start + 0.02)
        read(guard, 41, start + 0.05)
        read(guard, 41, start + 0.16)
    # A different joint has an independent budget.
    guard.check({"wrist_flex": 35}, 3)
    guard.check({"wrist_flex": -3}, 3.02)
    read(guard, 41, 3.03)
    with pytest.raises(HardwareError, match="recurring too often"):
        read(guard, -3, 3.04)


def test_old_episodes_leave_rolling_window():
    guard = TemperatureGuard(65, 0.5)
    for start in [0, 1, 2, 11]:
        read(guard, 41, start)
        read(guard, -3, start + 0.02)
        read(guard, 41, start + 0.05)
        assert read(guard, 41, start + 0.16)["temperature_feedback_pending"] == {}


def test_diagnostic_failure_cannot_extend_grace():
    def fail(*args, **kwargs):
        raise OSError("disk full")

    guard = TemperatureGuard(65, 0.5, fail)
    read(guard, 41, 0)
    read(guard, -3, 0.03)
    with pytest.raises(HardwareError, match="within 0.5s"):
        read(guard, -3, 0.5)


@pytest.mark.parametrize("grace", [-1, 1.01, True, "0.5", float("nan"), float("inf")])
def test_invalid_grace_config_is_rejected(cfg, grace):
    with pytest.raises(ValueError, match="temperature_feedback_grace_s"):
        validate_config(replace(cfg, robot=replace(cfg.robot, temperature_feedback_grace_s=grace)))


@pytest.mark.parametrize("grace", [0, 0.5, 1])
def test_valid_grace_config(cfg, grace):
    validate_config(replace(cfg, robot=replace(cfg.robot, temperature_feedback_grace_s=grace)))


@dataclass(frozen=True)
class State:
    status_code: int = 0
    pos: float = 0.25
    t_mos: float = 41


@pytest.fixture
def feedback_rig(cfg, monkeypatch):
    clock = [0.0]
    monkeypatch.setattr("wallx_client.hardware.b601.time.monotonic", lambda: clock[0])
    monkeypatch.setattr("wallx_client.hardware.b601.time.time", lambda: 100 + clock[0])
    states = dict.fromkeys(JOINTS, State())
    fb = Feedback.__new__(Feedback)
    fb.cfg, fb.error = cfg.robot, None
    fb.lock, fb.raw_frames = threading.Lock(), deque(maxlen=512)
    fb.last_sdk_states, fb.fault_recorded = (), False
    events = []
    fb.event = lambda kind, **kw: events.append((kind, kw))
    fb.temperature_guard = TemperatureGuard(65, 0.5, fb._temperature_event)
    fb.robot = SimpleNamespace(
        config=SimpleNamespace(motor_can_ids={name: (i, 253) for i, name in enumerate(JOINTS, 1)}),
        bus=SimpleNamespace(poll_feedback_once=lambda: None),
        motors={
            name: SignedTemperatureMotor(SimpleNamespace(get_state=lambda name=name: states[name]))
            for name in JOINTS
        },
    )

    def advance(now, value):
        clock[0] = now
        fb.latest = {i: (100 + now, 0) for i in range(1, 8)}
        states["shoulder_lift"] = State(t_mos=value)

    advance(0, 41)
    return fb, advance, events


def test_observation_keeps_raw_signed_value_and_positions(feedback_rig):
    fb, advance, events = feedback_rig
    before, _ = fb.snapshot()
    advance(0.02, 6550.6)  # SDK uint16 representation of FFE2.
    position, status = fb.snapshot()
    assert (position == before).all()
    assert status["motor_temperature_c"]["shoulder_lift"] == -3
    assert status["temperature_max_c"] == 41
    assert status["temperature_feedback_pending"]["shoulder_lift"]["reference_c"] == 41
    assert fb.error is None and not fb.fault_recorded
    assert events[0][0] == "temperature_feedback_transient"
    advance(0.05, 41)
    fb.snapshot()
    advance(0.151, 41)
    assert fb.snapshot()[1]["temperature_feedback_pending"] == {}
    assert events[-1][0] == "temperature_feedback_recovered"


@pytest.mark.parametrize("fault", ["stale", "can_flag", "sdk_status", "nonfinite_position"])
def test_temperature_grace_does_not_bypass_other_feedback_checks(feedback_rig, fault):
    fb, advance, events = feedback_rig
    fb.snapshot()
    advance(0.02, 6550.6)
    fb.snapshot()
    advance(0.04, 6550.6)
    if fault == "stale":
        fb.latest[1] = (99, 0)
    elif fault == "can_flag":
        fb.latest[1] = (100.04, 1)
    else:
        state = State(status_code=1) if fault == "sdk_status" else State(pos=float("nan"))
        fb.robot.motors[JOINTS[0]] = SimpleNamespace(get_state=lambda: state)
    with pytest.raises(HardwareError):
        fb.snapshot()
    assert fb.error is not None
    assert events[-1][0] == "motor_feedback_fault"


def test_driver_and_observation_checks_share_the_deadline(feedback_rig):
    fb, advance, events = feedback_rig
    fb.snapshot()
    advance(0.02, 6550.6)
    fb.check_temperatures({"shoulder_lift": -3})  # Native read entry point.
    advance(0.3, 6533.6)
    fb.snapshot()
    advance(0.5, 6533.6)
    with pytest.raises(HardwareError, match="within 0.5s"):
        fb.snapshot()
    advance(0.6, 41)
    with pytest.raises(HardwareError):
        fb.refresh()  # Fault stays latched; no CAN requests or homing.
    assert events[-1][0] == "motor_feedback_fault"
