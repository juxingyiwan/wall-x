"""Euler sampling independent of model context, embedding and KV-cache layout."""

from typing import Protocol

from .rtc import guided_velocity
from .rtc_contract import prepare_condition, validate_rtc_times


class FlowAdapter(Protocol):
    # Denoise returns dx/dt in this convention, including for x-pred models.
    time_convention: str

    def denoise(self, t, x): ...

    def after_step(self, index): ...


def sample_flow(adapter: FlowAdapter, x, times, mask, condition=None):
    if condition is not None:
        validate_rtc_times(times, getattr(adapter, "time_convention", None))
        condition = prepare_condition(condition, x)
    trajectory = []
    for index, t in enumerate(times[:-1]):
        # The callback must execute inside guided_velocity's gradient context.
        # Passing a precomputed velocity would silently break full PI VJP.
        velocity = (
            adapter.denoise(t, x)
            if condition is None
            else guided_velocity(adapter.denoise, x, t, condition, mask)
        )
        x = (x + (times[index + 1] - t) * velocity).detach()
        trajectory.append(x)
        adapter.after_step(index)
    return trajectory
