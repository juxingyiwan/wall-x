"""Offline wire/SDK compatibility checks; no CAN controller is created."""

import math
import struct
from dataclasses import dataclass, replace
from types import SimpleNamespace

import numpy as np
import pytest

from wallx_client.contracts import HardwareError
from wallx_client.hardware.robstride_temperature import (
    SignedTemperatureMotor,
    correct_sdk_temperature,
    decode_temperature_word,
)
from wallx_client.hardware.temperature_guard import TemperatureGuard


@pytest.mark.parametrize(
    "word,expected",
    [
        (0x0000, 0),
        (0x01AE, 43),
        (0x028A, 65),
        (0x7FFF, 3276.7),
        (0x8000, -3276.8),
        (0xFF38, -20),
        (0xFF60, -16),
        (0xFFFF, -0.1),
    ],
)
def test_signed_big_endian_temperature_vectors(word, expected):
    assert decode_temperature_word(word) == expected


def test_every_sdk_float32_word_roundtrips_without_sign_or_rounding_loss():
    # Independent signed-byte oracle over all wire values, including sign edges.
    for word in range(65536):
        sdk_value = float(np.float32(word) * np.float32(0.1))
        expected = struct.unpack(">h", struct.pack(">H", word))[0] / 10.0
        actual = correct_sdk_temperature(sdk_value)
        assert abs(actual - expected) < 0.001, (word, sdk_value, actual, expected)
        if word < 0x8000:
            assert actual == sdk_value  # Preserve normal positive readings exactly.
        assert correct_sdk_temperature(actual) == actual  # No double correction.


@pytest.mark.parametrize("value", [6553.6, 7000, 6533.64, 35.02])
def test_malformed_sdk_values_do_not_become_plausible_temperatures(value):
    with pytest.raises(HardwareError, match="malformed SDK value"):
        correct_sdk_temperature(value)


@pytest.mark.parametrize("value", [float("nan"), float("inf"), -float("inf")])
def test_nonfinite_values_are_rejected_by_validation(value):
    assert not math.isfinite(correct_sdk_temperature(value))
    with pytest.raises(HardwareError, match="feedback invalid"):
        TemperatureGuard(65, 0).check({"shoulder_lift": correct_sdk_temperature(value)}, 0)


@dataclass(frozen=True)
class MotorState:
    can_id: int = 2
    arbitration_id: int = 0x028002FD
    status_code: int = 0
    pos: float = 2.1940433979034424
    vel: float = 0.12970566749572754
    torq: float = -11.017426490783691
    t_mos: float = 6533.60009765625
    t_rotor: float = 0


def test_get_state_corrects_real_capture_and_preserves_every_other_field_and_command():
    original = MotorState()
    calls = []
    handle = SimpleNamespace(
        get_state=lambda: original,
        send_mit=lambda *args: calls.append(args),
        close=lambda: calls.append("close"),
    )
    motor = SignedTemperatureMotor(handle)
    state = motor.get_state()
    assert state.t_mos == -20.0
    assert replace(state, t_mos=original.t_mos) == original
    assert original.t_mos == 6533.60009765625
    assert motor.last_sdk_state is original
    assert state is not original
    motor.send_mit(1, 2, 3, 4, 5)
    motor.close()
    assert calls == [(1, 2, 3, 4, 5), "close"]
    with pytest.raises(HardwareError, match="decoded=-20.0°C"):
        TemperatureGuard(65, 0).check({"shoulder_lift": state.t_mos}, 0)


def test_positive_and_missing_states_are_unchanged():
    original = MotorState(t_mos=43.0)
    handle = SimpleNamespace(get_state=lambda: original)
    motor = SignedTemperatureMotor(handle)
    assert motor.get_state() is original
    handle.get_state = lambda: None
    assert motor.get_state() is None and motor.last_sdk_state is None
