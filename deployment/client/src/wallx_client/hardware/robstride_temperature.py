"""Legacy signed temperature interpretation at the B601 MotorBridge boundary.

MotorBridge 0.5.3 decodes data[6:8] as big-endian uint16 * 0.1:
https://github.com/motorbridge/motorbridge/blob/v0.5.3/motor_vendors/robstride/src/protocol.rs
Recover that word before interpreting its sign, rather than subtracting 6553.6
from an imprecise float. This adapter is scoped to B601 RobStride motor handles.
It neither patches MotorBridge globally nor changes position/velocity/torque.

Signedness is NOT established by the vendor feedback-frame table. The vendor
Python/C examples also use uint16; signed parameter registers are a different
interface. Preserve this interpretation for existing logs and numeric
compatibility, not as proof of a real subzero temperature or a corrected SDK defect.
TemperatureGuard validates the interpreted values against configured bounds.
"""

import math
from dataclasses import replace

from ..contracts import HardwareError


def decode_temperature_word(word):
    """Decode the complete 16-bit two's-complement word in tenths of a degree."""
    if type(word) is not int or not 0 <= word <= 0xFFFF:
        raise ValueError("RobStride temperature requires a 16-bit word")
    return (word - 0x10000 if word & 0x8000 else word) / 10.0


def correct_sdk_temperature(value):
    """Undo the pinned SDK's unsigned decode; an already signed value is unchanged."""
    if not math.isfinite(value) or value < 0:
        return value  # Validation remains separate from numeric decoding.
    word = round(value * 10)
    # The pinned Rust code returns f32, whose error here is < 0.001 degrees.
    # Reject malformed/off-grid values instead of truncating them into safe data.
    if not 0 <= word <= 0xFFFF or abs(value - word / 10.0) > 0.001:
        raise HardwareError(f"motor temperature feedback invalid; malformed SDK value={value}")
    return decode_temperature_word(word) if word & 0x8000 else value


class SignedTemperatureMotor:
    """Delegate commands unchanged and return a corrected immutable state snapshot."""

    def __init__(self, motor):
        self._motor = motor
        self.last_sdk_state = None

    def __getattr__(self, name):
        return getattr(self._motor, name)

    def get_state(self):
        state = self._motor.get_state()
        self.last_sdk_state = state
        if state is None:
            return None
        temperature = correct_sdk_temperature(state.t_mos)
        if temperature == state.t_mos or not math.isfinite(temperature):
            return state
        return replace(state, t_mos=temperature)
