"""Configured temperature bounds and bounded tolerance for below-minimum feedback."""

import math
from collections import deque

from ..contracts import HardwareError

RECOVERY_S = 0.1
HEADROOM_C = 5.0
EPISODE_WINDOW_S = 10.0
MAX_EPISODES = 3


class TemperatureFeedbackInvalid(HardwareError):
    """Unusable finite feedback; stop the episode, revalidate only for shutdown."""


class TemperatureLimitExceeded(HardwareError):
    """The configured temperature limit was reached; homing stays inhibited."""


class TemperatureGuard:
    """One control-thread guard shared by observation and native driver reads.

    Never smooth high readings or replace the measured temperature. A below-minimum
    sample may use a recent healthy reference for a bounded grace period. The
    deadline is anchored to the last healthy read and cannot be extended by
    alternating good/bad samples or by calling through another adapter path.
    """

    def __init__(self, limit, grace_s, event=None, *, minimum=0.0):
        self.limit = limit
        self.minimum = minimum
        self.grace_s = grace_s
        self.event = event
        self.good = {}
        self.pending = {}
        self.starts = {}
        self.error = None
        self.last_checked = None

    def _emit(self, kind, **fields):
        if self.event is not None:
            try:
                self.event(kind, **fields)
            except Exception:
                pass  # Diagnostics must not change protection or cleanup.

    def check(self, temperatures, now):
        if self.error is not None:
            raise self.error
        try:
            return self._check(temperatures, now)
        except HardwareError as exc:
            self.error = exc
            raise

    def _check_hard_faults(self, temperatures, now):
        if not math.isfinite(now) or (self.last_checked is not None and now < self.last_checked):
            raise HardwareError("Motor temperature monitor clock invalid")
        self.last_checked = now
        # Inspect every joint for hard faults before admitting any grace period.
        for name, value in temperatures.items():
            if not math.isfinite(value):
                raise HardwareError(f"{name}: motor temperature feedback invalid; decoded={value}")
            if value >= self.limit:
                raise TemperatureLimitExceeded(f"{name}: motor temperature {value}°C")

    def check_for_shutdown(self, temperatures, now):
        """Strict fresh measurements only; never clear the policy fault latch.

        The caller verifies seven-motor freshness and a continuous recovery
        window. No cached reference or ordinary grace may authorize a hold/home.
        A below-minimum sample restarts that window; any hard fault ends recovery.
        """
        if self.error is not None and not isinstance(self.error, TemperatureFeedbackInvalid):
            raise self.error
        try:
            self._check_hard_faults(temperatures, now)
            for name, value in temperatures.items():
                if value > self.limit - HEADROOM_C:
                    raise HardwareError(f"{name}: shutdown temperature too close to limit")
            for name, value in temperatures.items():
                if value < self.minimum:
                    self._invalid(name, value, "shutdown requires valid temperature feedback")
        except TemperatureFeedbackInvalid:
            raise
        except HardwareError as exc:
            self.error = exc
            raise
        return {"temperature_max_c": max(temperatures.values()), "temperature_feedback_pending": {}}

    def _check(self, temperatures, now):
        self._check_hard_faults(temperatures, now)
        for name, value in temperatures.items():
            episode = self.pending.get(name)
            if value < self.minimum:
                good = self.good.get(name)
                if episode is None:
                    if good is None:
                        self._invalid(name, value, "no healthy temperature reference")
                    reference, checked = good
                    if self.grace_s == 0 or now - checked >= self.grace_s:
                        self._invalid(name, value, "healthy reference expired or grace disabled")
                    if reference > self.limit - HEADROOM_C:
                        self._invalid(
                            name, value, "healthy reference too close to temperature limit"
                        )
                    starts = self.starts.setdefault(name, deque())
                    while starts and now - starts[0] >= EPISODE_WINDOW_S:
                        starts.popleft()
                    if len(starts) >= MAX_EPISODES:
                        self._invalid(
                            name,
                            value,
                            "low temperature feedback recurring too often (3 episodes/10s)",
                        )
                    starts.append(now)
                    episode = dict(since=checked, recovering_since=None, negative_reads=0)
                    self.pending[name] = episode
                    self._emit(
                        "temperature_feedback_transient",
                        joint=name,
                        decoded_c=value,
                        reference_c=reference,
                        reference_age_s=now - checked,
                        grace_s=self.grace_s,
                    )
                episode["recovering_since"] = None
                episode["negative_reads"] += 1
            if episode is not None:
                if now - episode["since"] >= self.grace_s:
                    self._invalid(name, value, f"did not recover within {self.grace_s:g}s")
                if value >= self.minimum:
                    if value > self.limit - HEADROOM_C:
                        self._invalid(name, value, "recovery too close to temperature limit")
                    if episode["recovering_since"] is None:
                        episode["recovering_since"] = now
                    if now - episode["recovering_since"] >= RECOVERY_S:
                        del self.pending[name]
                        self.good[name] = (value, now)
                        self._emit(
                            "temperature_feedback_recovered",
                            joint=name,
                            decoded_c=value,
                            elapsed_s=now - episode["since"],
                            negative_reads=episode["negative_reads"],
                        )
            else:
                self.good[name] = (value, now)
        # Raw readings are logged separately. These references are only for
        # monitoring during grace, never synthetic observations or motor inputs.
        references = {
            name: max(self.good[name][0], value) if name in self.pending else value
            for name, value in temperatures.items()
        }
        return {
            "temperature_max_c": max(references.values()),
            "temperature_feedback_pending": {
                name: dict(reference_c=self.good[name][0], age_s=now - episode["since"])
                for name, episode in self.pending.items()
            },
        }

    @staticmethod
    def _invalid(name, value, reason):
        raise TemperatureFeedbackInvalid(
            f"{name}: motor temperature feedback invalid; decoded={value:.1f}°C; {reason}"
        )
