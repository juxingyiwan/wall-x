"""Bounded per-camera timing counters, without retaining image data."""

from collections import deque
from dataclasses import dataclass


class CaptureClock:
    """Conservative capture bound that includes the driver's pending buffers.

    OpenCV's V4L2 backend requeues a buffer after retrieve/decode. With N
    buffers, a dequeued image may have been captured before the current grab.
    Its buffer was previously dequeued N reads ago and could only be refilled
    after that grab completed. Use that completion as the lower bound, not
    the beginning of a possibly blocking grab. The first N frames keep the
    stream-start bound, and a reader stall never makes queued frames fresh.
    """

    def __init__(self, buffers):
        self.dequeued = deque(maxlen=buffers)
        self.stream_start = None

    def start(self, now):
        if self.stream_start is None:
            self.stream_start = now
        if len(self.dequeued) < self.dequeued.maxlen:
            return self.stream_start
        return self.dequeued[0]

    def completed(self, grabbed):
        self.dequeued.append(grabbed)


@dataclass
class CameraMetrics:
    frames: int = 0
    first_grab: float | None = None
    last_grab: float | None = None
    reading_since: float | None = None
    grab_total_s: float = 0.0
    grab_max_s: float = 0.0
    decode_total_s: float = 0.0
    decode_max_s: float = 0.0
    interval_max_s: float = 0.0
    error: str | None = None

    def completed(self, started, grabbed, decoded):
        self.frames += 1
        if self.last_grab is not None:
            self.interval_max_s = max(self.interval_max_s, grabbed - self.last_grab)
        if self.first_grab is None:
            self.first_grab = grabbed
        self.last_grab = grabbed
        self.reading_since = None
        self.grab_total_s += grabbed - started
        self.grab_max_s = max(self.grab_max_s, grabbed - started)
        self.decode_total_s += decoded - grabbed
        self.decode_max_s = max(self.decode_max_s, decoded - grabbed)

    def summary(self, now):
        span = 0 if self.first_grab is None else self.last_grab - self.first_grab
        return {
            "frames": self.frames,
            "measured_fps": (self.frames - 1) / span if span > 0 else None,
            "grab_mean_s": self.grab_total_s / self.frames if self.frames else None,
            "grab_max_s": self.grab_max_s,
            "decode_mean_s": self.decode_total_s / self.frames if self.frames else None,
            "decode_max_s": self.decode_max_s,
            "interval_max_s": self.interval_max_s,
            "current_read_elapsed_s": None
            if self.reading_since is None
            else now - self.reading_since,
            "last_grab_age_s": None if self.last_grab is None else now - self.last_grab,
            "error": self.error,
        }
