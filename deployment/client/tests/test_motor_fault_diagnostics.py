"""Keep temperature faults distinct from operator stops and retain raw evidence."""

import json
import sys
import threading
from collections import deque
from dataclasses import dataclass
from types import SimpleNamespace

import numpy as np
import pytest

from wallx_client.contracts import JOINTS, HardwareError
from wallx_client.hardware.b601 import B601, Feedback
from wallx_client.hardware.robstride_temperature import (
    SignedTemperatureMotor,
    correct_sdk_temperature,
)
from wallx_client.hardware.temperature_guard import TemperatureGuard


@dataclass(frozen=True)
class State:
    status_code: int = 0
    pos: float = 0.0
    t_mos: float = 35.0


def trace(cfg, event):
    feedback = Feedback.__new__(Feedback)
    feedback.cfg = cfg.robot
    feedback.event = event
    feedback.fault_recorded = False
    feedback.lock = threading.Lock()
    feedback.raw_frames = deque(maxlen=512)
    feedback.last_sdk_states = ()
    feedback.error = None
    feedback.temperature_guard = TemperatureGuard(65, 0.5, feedback._temperature_event)
    return feedback


@pytest.mark.parametrize("temperature", [65, 100, -20, -16, float("nan"), float("inf")])
def test_temperature_guard_does_not_discard_implausible_readings(temperature):
    with pytest.raises(HardwareError, match="shoulder_lift: motor temperature"):
        TemperatureGuard(65, 0).check({"shoulder_lift": temperature}, 0)
    TemperatureGuard(65, 0).check({"shoulder_lift": 64.9}, 0)


@pytest.mark.parametrize("temperature", [65, 100, 135])
def test_actual_overtemperature_still_uses_existing_threshold(temperature):
    with pytest.raises(HardwareError) as caught:
        TemperatureGuard(65, 0).check({"shoulder_lift": temperature}, 0)
    assert "feedback invalid" not in str(caught.value)


def test_captured_motor2_feedback_is_invalid_and_remains_latched(cfg, monkeypatch):
    # Actual type-2 response at 2026-10-08 14:19:01.659 +08:00.
    # The independent listener also captured type-24 packets with the same word.
    raw = bytes.fromhex("9658805458d3ff38")
    sdk_temperature = float(np.float32(int.from_bytes(raw[6:8], "big") * 0.1))
    events = []
    feedback = trace(cfg, lambda kind, **kw: events.append((kind, kw)))
    feedback.error = None
    timestamp = 1791440341.65934
    monkeypatch.setattr("wallx_client.hardware.b601.time.time", lambda: timestamp + 0.01)
    states = {name: State() for name in JOINTS}
    states["shoulder_lift"] = State(t_mos=sdk_temperature)
    feedback.robot = SimpleNamespace(
        config=SimpleNamespace(motor_can_ids={name: (i, 0xFD) for i, name in enumerate(JOINTS, 1)}),
        bus=SimpleNamespace(poll_feedback_once=lambda: None),
        motors={
            name: SignedTemperatureMotor(SimpleNamespace(get_state=lambda name=name: states[name]))
            for name in JOINTS
        },
    )
    feedback.latest = {i: (timestamp, 0) for i in range(1, 8)}
    feedback._remember_frame(
        SimpleNamespace(
            is_rx=True,
            timestamp=timestamp,
            arbitration_id=0x028002FD,
            is_extended_id=True,
            is_error_frame=False,
            dlc=8,
            data=raw,
        )
    )
    with pytest.raises(HardwareError, match="temperature feedback invalid") as caught:
        feedback.snapshot()
    message = str(caught.value)
    assert "decoded=-20.0°C" in message and "no healthy temperature reference" in message
    assert "6533" not in message
    assert events[0][1]["recent_can_frames"][0]["data_hex"] == raw.hex()
    assert events[0][1]["last_snapshot_states"]["shoulder_lift"]["t_mos"] == -20.0
    assert events[0][1]["sdk_unsigned_temperatures_c"]["shoulder_lift"] == sdk_temperature
    assert events[0][1]["fault_type"] == "TemperatureFeedbackInvalid"
    assert events[0][1]["temperature_signedness_verified"] is False
    # A subsequent healthy reading must not clear the policy fault latch.
    states["shoulder_lift"] = State(t_mos=43)
    with pytest.raises(HardwareError, match="temperature feedback invalid"):
        feedback.snapshot()
    assert len(events) == 1


def test_fault_trace_is_bounded_timestamped_and_written_once(cfg):
    events = []
    feedback = trace(cfg, lambda kind, **kw: events.append((kind, kw)))
    for sequence in range(600):
        feedback._remember_frame(
            SimpleNamespace(
                is_rx=True,
                timestamp=sequence,
                arbitration_id=0x020002FD,
                is_extended_id=True,
                is_error_frame=False,
                dlc=8,
                data=bytes.fromhex("7ffd802b7fff0154"),
            )
        )
    feedback.last_sdk_states = (SimpleNamespace(t_mos=float("nan")),)
    feedback.record_fault("test", native_temperatures={"shoulder_lift": 6537.6})
    feedback.record_fault("later")
    assert len(events) == 1
    kind, payload = events[0]
    assert kind == "motor_feedback_fault"
    assert len(payload["recent_can_frames"]) == 512
    assert payload["recent_can_frames"][0]["timestamp"] == 88
    assert payload["recent_can_frames"][-1]["data_hex"] == "7ffd802b7fff0154"
    assert payload["native_temperatures_c"]["shoulder_lift"] == 6537.6
    assert payload["last_snapshot_states"][JOINTS[0]]["t_mos"] == "nan"
    json.dumps(payload, allow_nan=False)


def test_failed_diagnostic_writer_does_not_replace_motor_fault(cfg):
    def broken(*args, **kw):
        raise OSError("disk full")

    feedback = trace(cfg, broken)
    feedback.record_fault("motor temperature")
    assert feedback.fault_recorded


@pytest.mark.parametrize("temperature", [65, 105, 6537.6, 6533.60009765625])
@pytest.mark.parametrize("warm_reference", [False, True])
def test_native_temperature_read_latches_fault_before_driver_interrupt(
    cfg, monkeypatch, temperature, warm_reference
):
    import wallx_client.hardware.b601 as module

    sent, diagnostics = [], []
    temperatures = {"shoulder_lift": temperature}

    class Driver:
        def __init__(self, config):
            self.config = config
            self.motors = {
                "shoulder_lift": SimpleNamespace(
                    get_state=lambda: State(t_mos=temperatures["shoulder_lift"])
                )
            }

        def _add_motors_to_bus(self):
            pass

        def _read_motor_temperatures(self):
            return {name: motor.get_state().t_mos for name, motor in self.motors.items()}

        def send_action(self, action):
            if max(self._read_motor_temperatures().values()) > 100:
                raise KeyboardInterrupt("Overheat detected")
            sent.append(action)
            return action

    def config(**kw):
        return SimpleNamespace(
            max_relative_target=None,
            joint_directions=dict.fromkeys(JOINTS, 1),
            joint_limits=dict.fromkeys(JOINTS, (-500, 500)),
            **kw,
        )

    guard = TemperatureGuard(65, 0.5)
    clock = [0.02]
    if warm_reference:
        guard.check({"shoulder_lift": 41}, 0)
    feedback = SimpleNamespace(
        check_temperatures=lambda temperatures: guard.check(temperatures, clock[0]),
        refresh=lambda: None,
        record_fault=lambda *a, **kw: diagnostics.append((a, kw)),
    )
    monkeypatch.setitem(
        sys.modules,
        "lerobot_robot_seeed_b601",
        SimpleNamespace(
            SeeedB601RSFollower=Driver,
            SeeedB601RSFollowerConfig=config,
        ),
    )
    monkeypatch.setitem(
        sys.modules, "motorbridge", SimpleNamespace(Controller=lambda **kw: object())
    )
    monkeypatch.setattr(module, "Feedback", lambda *a: feedback)
    monkeypatch.setattr(
        module,
        "Cameras",
        lambda *a, **kw: SimpleNamespace(
            latest=True,
            error=None,
            snapshot=lambda: None,
        ),
    )
    hw = B601(cfg.robot, lambda *a, **kw: None)
    hw.open_observation()
    if warm_reference and correct_sdk_temperature(temperature) < 0:
        hw.send(np.zeros(7))
        assert len(sent) == 1
        assert diagnostics == []
        assert hw._home_inhibited is None
        sent.clear()
        clock[0] = 0.5
    with pytest.raises(HardwareError, match="shoulder_lift: motor temperature"):
        hw.send(np.zeros(7))
    assert sent == []
    assert len(diagnostics) == 1
    assert diagnostics[0][1]["native_temperatures"] == {
        "shoulder_lift": correct_sdk_temperature(temperature)
    }
    if correct_sdk_temperature(temperature) < 0:
        assert hw._home_inhibited is None
        from wallx_client.hardware.temperature_guard import TemperatureFeedbackInvalid

        assert isinstance(feedback.error, TemperatureFeedbackInvalid)
    else:
        assert hw._home_inhibited is not None
        temperatures["shoulder_lift"] = 34
        with pytest.raises(HardwareError, match="shoulder_lift: motor temperature"):
            hw._home_feedback()  # A real overtemperature latch stays permanent.
