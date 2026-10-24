"""Bounded JSONL logging, source RGB snapshots and predicted action chunks."""

import json
import queue
import threading
import time
import uuid
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

import numpy as np

from . import __version__


def write_json(path: Path, value: dict) -> None:
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    temp.replace(path)


class Recorder:
    def __init__(self, config, mode: str, output: str | None = None):
        self.config = config
        self.telemetry = None
        run_id = datetime.now().strftime("%Y%m%d_%H%M%S") + "_" + uuid.uuid4().hex[:8]
        self.path = (
            Path(output).expanduser() if output else Path(config.output_root).expanduser() / run_id
        )
        self.path.mkdir(parents=True, exist_ok=False)
        self.queue = queue.Queue(maxsize=2048)
        self.error = None
        write_json(
            self.path / "config.json",
            {"client_version": __version__, "mode": mode, "configuration": asdict(config)},
        )
        self.thread = threading.Thread(target=self._write_loop, name="rollout-log", daemon=True)
        self.thread.start()

    def start_telemetry(self):
        if self.config.display_data or self.config.save_rollout:
            from .telemetry import Telemetry

            self.telemetry = Telemetry(self.config, self.path)

    def sample(self, observation, **kwargs):
        if self.telemetry:
            self.telemetry.publish(observation, **kwargs)

    def check_media(self):
        if self.telemetry:
            self.telemetry.check()

    def _write_loop(self):
        try:
            with (
                (self.path / "events.jsonl").open("w", encoding="utf-8", buffering=1) as events,
                (self.path / "commands.jsonl").open("w", encoding="utf-8", buffering=1) as commands,
            ):
                while True:
                    entry = self.queue.get()
                    if entry is None:
                        return
                    target, row = entry
                    if target == "prediction":
                        self.prediction(row)
                        continue
                    (commands if target == "command" else events).write(
                        json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n"
                    )
        except Exception as exc:
            self.error = exc

    def append(self, target: str, **row):
        if self.error:
            raise OSError(f"Rollout logger failed: {self.error}")
        row = {"wall_time": time.time(), "monotonic": time.monotonic(), **row}
        try:
            self.queue.put_nowait((target, row))
        except queue.Full as exc:
            raise OSError("Rollout log queue is full; control cannot continue unrecorded") from exc

    def event(self, kind: str, **row):
        self.append("event", kind=kind, **row)

    def prediction(self, result):
        # Synchronous for preflight/RTC; ordinary control queues this work.
        obs = result.observation
        np.savez_compressed(
            self.path / f"query_{result.request_id}.npz",
            state=obs.state,
            **obs.views,
            actions=result.actions,
        )

        write_json(
            self.path / f"query_{result.request_id}.json",
            {
                "request_id": result.request_id,
                "generation": result.generation,
                "captured_monotonic": obs.captured_monotonic,
                "captured_wall": obs.captured_wall,
                "state_monotonic": obs.state_monotonic,
                "camera_skew_s": obs.camera_skew_s,
                "capture_sequence": obs.capture_sequence,
                "camera_generation": obs.camera_generation,
                "roundtrip_s": result.roundtrip_s,
                "server_timing": result.server_timing,
                "policy_timing": (
                    {
                        "preparation_started_monotonic": result.timing.preparation_started_monotonic,
                        "dispatched_monotonic": result.timing.dispatched_monotonic,
                        "completed_monotonic": result.timing.completed_monotonic,
                        "preparation_s": result.timing.preparation_s,
                        "observation_to_result_s": result.observation_to_result_s,
                    }
                    if result.timing is not None
                    else None
                ),
                "rtc": result.rtc,
            },
        )

    def prediction_async(self, result):
        """Queue an owned prediction snapshot without delaying action admission."""
        if self.error:
            raise OSError(f"Rollout logger failed: {self.error}") from self.error
        try:
            self.queue.put_nowait(("prediction", result))
        except queue.Full as exc:
            raise OSError("Rollout log queue is full; prediction cannot be recorded") from exc

    def observation(self, obs, stem):
        """Save an operator-readable pair during capture/preflight only."""
        import cv2

        self.sample(obs)

        np.savez_compressed(self.path / f"{stem}.npz", state=obs.state, **obs.views)
        for name, rgb in obs.views.items():
            path = self.path / f"{stem}_{name}.png"
            if not cv2.imwrite(str(path), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)):
                raise OSError(f"Cannot write camera preview: {path}")

    def close(self, report: dict):
        if self.telemetry:
            media = self.telemetry.close()
            report = {**report, "rollout_media": media}
            if media["error"]:
                self.error = OSError(media["error"])
        if self.thread.is_alive():
            try:
                self.queue.put(None, timeout=3)
                self.thread.join(timeout=5)
            except queue.Full:
                self.error = OSError("Log queue did not drain")
        if self.thread.is_alive():
            self.error = OSError("Log writer did not terminate")
        write_json(
            self.path / "report.json",
            {**report, "log_error": str(self.error) if self.error else None},
        )
        if self.error:
            raise OSError(f"Rollout logger did not finish cleanly: {self.error}")
