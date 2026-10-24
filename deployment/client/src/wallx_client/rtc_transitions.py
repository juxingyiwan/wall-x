"""Clock-free session decisions; the caller performs I/O and emits notices.

Sessions are single-owner objects. Workers receive request snapshots and return
values; they never mutate sessions. No future, thread, hardware or logger is
accepted by this interface. Every scheduling timestamp comes from the caller.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Generic, TypeVar

from .lifecycle import EpisodeBudget


class Decision(str, Enum):
    WAIT = "wait"
    HOLD = "hold"
    REQUEST = "request"
    DISCARD = "discard"
    ACTIVATE = "activate"
    COMMAND = "command"
    FAULT = "fault"
    STOP = "stop"


@dataclass(frozen=True)
class Notice:
    name: str
    values: dict


@dataclass
class Effects:
    decision: Decision = Decision.WAIT
    notices: list[Notice] = field(default_factory=list)

    def note(self, name, **values):
        self.notices.append(Notice(name, values))


RequestT = TypeVar("RequestT")


class SessionControl(Generic[RequestT]):
    """Common request ownership; time/index policy belongs to each strategy."""

    def __init__(self, cfg, started):
        self.cfg = cfg
        self.started = self.next_query = started
        self.tick_s = 1 / cfg.control.hz
        self.generation = self.plan_id = 0
        self.paused = self.stopped = False
        self.pending: RequestT | None = None
        self.active_query = None
        self.episode_budget = EpisodeBudget(cfg.control, started)
        self.metrics = {}

    def ensure_live(self):
        if self.stopped:
            raise RuntimeError("RTC session has stopped")

    def own_request(self, context: RequestT) -> RequestT:
        self.ensure_live()
        if self.pending is not None:
            raise RuntimeError("RTC session already owns an in-flight request")
        self.pending = context
        return context

    def take_request(self) -> RequestT:
        self.ensure_live()
        if self.pending is None:
            raise RuntimeError("RTC result has no in-flight request")
        context, self.pending = self.pending, None
        return context

    def stop(self):
        if not self.stopped:
            self.generation += 1
            self.stopped = True
            self.pending = None
        return Effects(Decision.STOP)
