"""Bound the blocking native B601 homing routine on the Linux control thread."""

import signal
import threading
from contextlib import contextmanager

from ..contracts import HardwareError


class HomingTimeout(BaseException):
    """Bypass driver Exception handlers; the adapter still disables in finally."""


@contextmanager
def homing_deadline(seconds):
    # A thread timeout cannot stop a worker that still owns the CAN connection.
    # Keep native homing in the motor owner thread and interrupt its Python I/O.
    # An uninterruptible kernel/extension hang still requires hardware protection.
    if threading.current_thread() is not threading.main_thread() or not hasattr(
        signal, "setitimer"
    ):
        raise HardwareError("Bounded B601 homing requires the Linux main thread")
    if any(signal.getitimer(signal.ITIMER_REAL)):
        raise HardwareError("Cannot replace an active process alarm during homing")

    def expired(*_):
        raise HomingTimeout(f"Native homing exceeded {seconds:g}s")

    previous = signal.signal(signal.SIGALRM, expired)
    try:
        signal.setitimer(signal.ITIMER_REAL, seconds)
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, previous)
