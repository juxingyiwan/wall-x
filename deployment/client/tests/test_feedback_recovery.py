"""Shutdown-only feedback recovery; fake clock, reports and motors, no CAN I/O."""

import threading
from collections import deque
from types import SimpleNamespace

import numpy as np
import pytest

from wallx_client.contracts import JOINTS, HardwareError
from wallx_client.hardware.b601 import Feedback, StaleFeedback
from wallx_client.hardware.temperature_guard import (
    TemperatureFeedbackInvalid,
    TemperatureGuard,
    TemperatureLimitExceeded,
)


def feedback_rig(cfg, monkeypatch):
    clock = [1000.0]
    monkeypatch.setattr("wallx_client.hardware.b601.time.time", lambda: clock[0])
    monkeypatch.setattr("wallx_client.hardware.b601.time.monotonic", lambda: clock[0])
    monkeypatch.setattr(
        "wallx_client.hardware.b601.time.sleep", lambda dt: clock.__setitem__(0, clock[0] + dt)
    )
    fb = Feedback.__new__(Feedback)
    fb.cfg = cfg.robot
    fb.error = StaleFeedback("Stale CAN feedback")
    fb.receive_error = None
    fb.lock = threading.Lock()
    fb.latest = {i: (999.0, 0, 350) for i in range(1, 8)}
    fb.last_sdk_states = ()
    fb.raw_frames = deque(maxlen=512)
    fb.event = None
    fb.temperature_guard = TemperatureGuard(65, 0.5)
    requests = []
    states = {j: SimpleNamespace(status_code=0, pos=0.1, t_mos=35) for j in JOINTS}

    def report(i):
        requests.append(i)
        word = round(states[JOINTS[i - 1]].t_mos * 10) & 0xFFFF
        fb.latest[i] = (clock[0], 0, word)

    fb.robot = SimpleNamespace(
        config=SimpleNamespace(motor_can_ids={j: (i, 0xFD) for i, j in enumerate(JOINTS, 1)}),
        bus=SimpleNamespace(poll_feedback_once=lambda: None),
        motors={
            j: SimpleNamespace(
                get_state=lambda j=j: states[j],
                robstride_set_active_report=lambda _, i=i: report(i),
            )
            for i, j in enumerate(JOINTS, 1)
        },
    )
    return fb, clock, requests, states


def test_stale_stays_latched_for_control_but_shutdown_requires_successive_new_replies(
    cfg, monkeypatch
):
    fb, clock, requests, _ = feedback_rig(cfg, monkeypatch)
    fb.latest = {i: (clock[0], 0) for i in range(1, 8)}
    with pytest.raises(StaleFeedback):
        fb.snapshot()
    holds = []
    measured, _ = fb.recover_for_shutdown(1, 0.25, lambda q: holds.append((clock[0], q)))
    assert clock[0] - 1000 >= 0.25
    assert len(holds) >= 2 and len(requests) == 7 * len(holds)
    assert fb.error is None
    np.testing.assert_array_equal(fb.snapshot()[0], measured)


@pytest.mark.parametrize("missing", [1, 7])
def test_missing_one_motor_never_holds_or_homes_even_if_other_six_recover(
    cfg, monkeypatch, missing
):
    fb, clock, _, _ = feedback_rig(cfg, monkeypatch)
    fb.robot.motors[JOINTS[missing - 1]].robstride_set_active_report = lambda _: None
    holds = []
    with pytest.raises(HardwareError, match="did not recover"):
        fb.recover_for_shutdown(0.5, 0.1, holds.append)
    assert not holds
    assert 0.5 <= clock[0] - 1000 < 0.51
    assert not isinstance(fb.error, StaleFeedback)


def test_cached_fresh_enough_samples_cannot_count_as_new_recovery_replies(cfg, monkeypatch):
    fb, clock, _, _ = feedback_rig(cfg, monkeypatch)
    fb.latest = {i: (clock[0] - 0.01, 0) for i in range(1, 8)}
    for motor in fb.robot.motors.values():
        motor.robstride_set_active_report = lambda _: None
    with pytest.raises(HardwareError, match="did not recover"):
        fb.recover_for_shutdown(0.05, 0.01, lambda _: pytest.fail("cached feedback admitted"))


def test_flapping_replies_do_not_accumulate_a_success_window(cfg, monkeypatch):
    fb, clock, _, _ = feedback_rig(cfg, monkeypatch)
    count = [0]

    def intermittent(_):
        count[0] += 1
        if count[0] % 2:
            fb.latest[7] = (clock[0], 0, 350)

    fb.robot.motors[JOINTS[-1]].robstride_set_active_report = intermittent
    holds = []
    with pytest.raises(HardwareError, match="did not recover"):
        fb.recover_for_shutdown(0.6, 0.2, holds.append)
    assert holds  # Some healthy rounds are insufficient for continuous recovery.


def test_excessive_gap_between_valid_rounds_resets_recovery(cfg, monkeypatch):
    fb, clock, _, _ = feedback_rig(cfg, monkeypatch)

    def base_sleep(dt):
        clock[0] += dt

    monkeypatch.setattr(
        "wallx_client.hardware.b601.time.sleep", lambda dt: base_sleep(max(0.13, dt))
    )
    with pytest.raises(HardwareError, match="did not recover"):
        fb.recover_for_shutdown(0.8, 0.2, lambda _: None)


@pytest.mark.parametrize("fault", ["motor", "temperature", "position", "timestamp", "receiver"])
def test_recovery_does_not_clear_hard_faults_or_relax_checks(cfg, monkeypatch, fault):
    fb, clock, _, states = feedback_rig(cfg, monkeypatch)
    if fault == "motor":
        states[JOINTS[1]].status_code = 1
    elif fault == "temperature":
        states[JOINTS[1]].t_mos = 65
    elif fault == "position":
        states[JOINTS[1]].pos = float("nan")
    elif fault == "timestamp":
        fb.robot.motors[JOINTS[-1]].robstride_set_active_report = lambda _: fb.latest.update(
            {7: (clock[0] + 1, 0)}
        )
    else:
        fb.receive_error = OSError("CAN disconnected")
    with pytest.raises(HardwareError):
        fb.recover_for_shutdown(1, 0.25, lambda _: pytest.fail("unsafe hold"))
    assert fb.error is not None


def test_receive_fault_during_recovery_hold_cannot_be_erased(cfg, monkeypatch):
    fb, _, _, _ = feedback_rig(cfg, monkeypatch)

    def hold(_):
        fb.receive_error = OSError("CAN read failed")

    with pytest.raises(HardwareError, match="CAN read failed"):
        fb.recover_for_shutdown(1, 0.25, hold)
    assert fb.receive_error is not None and fb.error is not None


def test_latched_hard_fault_prevents_even_recovery_telemetry_requests(cfg, monkeypatch):
    fb, _, requests, _ = feedback_rig(cfg, monkeypatch)
    fb.error = HardwareError("overtemperature")
    with pytest.raises(HardwareError, match="overtemperature"):
        fb.recover_for_shutdown(1, 0.25, lambda _: None)
    assert requests == []


def latch_temperature_fault(fb, clock, states):
    fb.error = None
    fb.latest = {i: (clock[0], 0, 350) for i in range(1, 8)}
    fb.snapshot()
    states[JOINTS[1]].t_mos = -20
    fb.latest[2] = (clock[0], 0, 0xFF38)
    clock[0] += 0.02
    fb.snapshot()  # Existing bounded transient tolerance is unchanged.
    clock[0] += 0.5
    fb.latest = {i: (clock[0], 0, 0xFF38 if i == 2 else 350) for i in range(1, 8)}
    with pytest.raises(TemperatureFeedbackInvalid):
        fb.snapshot()


def test_ff38_fault_can_recover_only_for_shutdown_after_successive_good_replies(cfg, monkeypatch):
    fb, clock, requests, states = feedback_rig(cfg, monkeypatch)
    latch_temperature_fault(fb, clock, states)
    states[JOINTS[1]].t_mos = 35
    with pytest.raises(TemperatureFeedbackInvalid):
        fb.snapshot()  # A single good SDK value cannot restart control.
    holds = []
    started = clock[0]
    _, status = fb.recover_for_shutdown(1, 0.25, lambda _: holds.append(clock[0]))
    assert clock[0] - started >= 0.25
    assert len(requests) == 7 * len(holds) and len(holds) >= 2
    assert status["temperature_feedback_pending"] == {}
    assert fb.shutdown_only
    # The ordinary policy checker remains faulted even after shutdown recovery.
    with pytest.raises(TemperatureFeedbackInvalid):
        fb.temperature_guard.check({j: 35 for j in JOINTS}, clock[0])


@pytest.mark.parametrize("sdk_temperature", [-20, 35])
def test_persistent_raw_ff38_never_uses_a_cached_healthy_sdk_value(
    cfg, monkeypatch, sdk_temperature
):
    fb, clock, _, states = feedback_rig(cfg, monkeypatch)
    latch_temperature_fault(fb, clock, states)
    states[JOINTS[1]].t_mos = sdk_temperature
    fb.robot.motors[JOINTS[1]].robstride_set_active_report = lambda _: fb.latest.update(
        {2: (clock[0], 0, 0xFF38)}
    )
    started = clock[0]
    with pytest.raises(HardwareError, match="did not recover"):
        fb.recover_for_shutdown(1, 0.25, lambda _: pytest.fail("invalid temperature hold"))
    assert 1 <= clock[0] - started < 1.01


def test_temperature_flapping_restarts_the_entire_recovery_window(cfg, monkeypatch):
    fb, clock, _, states = feedback_rig(cfg, monkeypatch)
    latch_temperature_fault(fb, clock, states)
    states[JOINTS[1]].t_mos = 35
    count = [0]

    def report(_):
        count[0] += 1
        fb.latest[2] = (clock[0], 0, 0xFF38 if count[0] % 3 == 0 else 350)

    fb.robot.motors[JOINTS[1]].robstride_set_active_report = report
    with pytest.raises(HardwareError, match="did not recover"):
        fb.recover_for_shutdown(1, 0.25, lambda _: None)


@pytest.mark.parametrize(
    "fault", ["sdk_hot", "raw_hot", "position", "receiver", "motor", "near_limit"]
)
def test_new_hard_fault_wins_over_pending_invalid_temperature(cfg, monkeypatch, fault):
    fb, clock, _, states = feedback_rig(cfg, monkeypatch)
    latch_temperature_fault(fb, clock, states)
    if fault == "sdk_hot":
        states[JOINTS[0]].t_mos = 65
    elif fault == "raw_hot":
        fb.robot.motors[JOINTS[0]].robstride_set_active_report = lambda _: fb.latest.update(
            {1: (clock[0], 0, 650)}
        )
    elif fault == "position":
        states[JOINTS[0]].pos = float("nan")
    elif fault == "receiver":
        fb.receive_error = OSError("CAN read failed")
    elif fault == "motor":
        states[JOINTS[0]].status_code = 1
    else:
        states[JOINTS[0]].t_mos = 61
    started = clock[0]
    with pytest.raises(HardwareError) as caught:
        fb.recover_for_shutdown(1, 0.25, lambda _: pytest.fail("unsafe hold"))
    assert not isinstance(caught.value, TemperatureFeedbackInvalid)
    assert clock[0] == started  # Hard faults do not wait out the recovery budget.
    if fault in ("sdk_hot", "raw_hot"):
        assert isinstance(caught.value, TemperatureLimitExceeded)


def test_latched_overtemperature_cannot_be_cleared_by_later_normal_temperature(cfg, monkeypatch):
    fb, clock, requests, states = feedback_rig(cfg, monkeypatch)
    fb.error = None
    fb.latest = {i: (clock[0], 0, 650 if i == 2 else 350) for i in range(1, 8)}
    states[JOINTS[1]].t_mos = 65
    with pytest.raises(TemperatureLimitExceeded):
        fb.snapshot()
    states[JOINTS[1]].t_mos = 35
    with pytest.raises(TemperatureLimitExceeded):
        fb.recover_for_shutdown(1, 0.25, lambda _: pytest.fail("overtemperature hold"))
    assert requests == []


def test_feedback_age_threshold_still_ends_the_episode(cfg, monkeypatch):
    fb, clock, _, _ = feedback_rig(cfg, monkeypatch)
    fb.error = None
    fb.latest = {i: (clock[0] - cfg.robot.max_feedback_age_s - 0.001, 0) for i in range(1, 8)}
    with pytest.raises(StaleFeedback, match="Stale CAN feedback"):
        fb.snapshot()
    assert isinstance(fb.error, StaleFeedback)


def test_pending_temperature_grace_is_not_shutdown_recovery(cfg, monkeypatch):
    fb, _, _, states = feedback_rig(cfg, monkeypatch)
    fb.temperature_guard.check({j: 35 for j in JOINTS}, 999.99)
    states[JOINTS[1]].t_mos = -3
    with pytest.raises(HardwareError, match="did not recover"):
        fb.recover_for_shutdown(0.2, 0.1, lambda _: pytest.fail("unverified temperature"))


def test_receive_error_has_precedence_over_a_stale_control_latch(cfg, monkeypatch):
    fb, _, _, _ = feedback_rig(cfg, monkeypatch)
    fb.receive_error = OSError("socketCAN error frame")
    with pytest.raises(HardwareError, match="socketCAN error frame"):
        fb.snapshot()
    assert not isinstance(fb.error, StaleFeedback)


def test_motor_status_fault_is_latched_by_receiver_before_later_good_frames(cfg, monkeypatch):
    fb, clock, requests, _ = feedback_rig(cfg, monkeypatch)
    fb.stop = threading.Event()
    messages = iter(
        [
            SimpleNamespace(
                is_rx=True,
                is_extended_id=True,
                is_error_frame=False,
                timestamp=clock[0],
                arbitration_id=0x028102FD,
                dlc=8,
                data=bytes(8),
            )
        ]
    )
    fb.bus = SimpleNamespace(recv=lambda **kw: next(messages))
    fb._receive()
    assert "fault bits=1" in str(fb.receive_error)
    assert fb.latest[2][1] == 1
    fb.latest[2] = (clock[0], 0)
    with pytest.raises(HardwareError, match="fault bits=1"):
        fb.recover_for_shutdown(1, 0.25, lambda _: pytest.fail("motor fault hold"))
    assert requests == []
