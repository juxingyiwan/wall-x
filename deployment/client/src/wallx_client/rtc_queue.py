"""Original-action FIFO for fixed-rate RTC; no hardware access.

Consumption-bounded merging and hold padding follow LeRobot b9cb121 (see
NOTICE.txt). Processing stays in the existing hardware adapter;
driver-clipped targets never become model prefixes.
"""

import math

import numpy as np

from .rtc import request


def elapsed_steps(seconds, dt):
    if not math.isfinite(seconds) or seconds < 0:
        raise ValueError("Invalid RTC queue elapsed time")
    return max(0, math.ceil(seconds / dt - 1e-9))


def merge_delay(elapsed, consumed):
    """Do not skip motion that the actor never consumed (including bootstrap).

    Equivalent to LeRobot's _check_and_resolve_delays for nonnegative counts.
    Wall-clock latency is still measured for guidance and scheduling. Holds do
    not advance the original-action index, and consumption is not an assertion
    that the hardware has physically reached its target.
    """
    if any(type(n) is not int or n < 0 for n in (elapsed, consumed)):
        raise ValueError("Invalid RTC queue merge counts")
    return min(elapsed, consumed)


class ActionQueue:
    """Owned only by the actor thread; the inference worker gets copied input."""

    def __init__(self, horizon):
        self.horizon = horizon
        self.clear()

    def clear(self):
        self.actions = np.empty((0, 7), dtype=np.float32)
        self.index = 0
        self.model_offset = 0

    def __len__(self):
        return len(self.actions) - self.index

    def leftover(self):
        return self.actions[self.index :].copy()

    def replace(self, actions, delay):
        actions = np.asarray(actions)
        if actions.shape != (self.horizon, 7) or not np.isfinite(actions).all():
            raise ValueError("Invalid RTC queue action chunk")
        if type(delay) is not int or delay < 0:
            raise ValueError("Invalid RTC queue delay")
        if delay >= self.horizon:
            return False  # A late prediction must not erase an existing queue.
        self.actions = actions[delay:].copy()
        self.index, self.model_offset = 0, delay
        return True

    def pop(self):
        if not len(self):
            return None
        index = self.model_offset + self.index
        value = self.actions[self.index].copy()
        self.index += 1
        return value, index


def guided_request(cfg, epoch, plan_id, leftover, estimated_steps, horizon):
    """Extend a short original prefix with the last target, as LeRobot does.

    Guidance covers the latency estimate plus the requested blend. Wire v1
    requires one tapered and one free point, so commitment caps at H-2. This
    protocol bound does not change measured latency or queue merge counts.
    """
    leftover = np.asarray(leftover)
    if leftover.shape != (len(leftover), 7) or not np.isfinite(leftover).all():
        raise ValueError("Invalid RTC leftover actions")
    if not 0 <= len(leftover) <= horizon:
        raise ValueError("RTC leftover exceeds action horizon")
    if not len(leftover):
        return None
    committed = min(max(1, estimated_steps), horizon - 2)
    end = min(committed + cfg.blend_steps, horizon - 1)
    return request(cfg, epoch, plan_id, leftover, committed, guidance_horizon=end)
