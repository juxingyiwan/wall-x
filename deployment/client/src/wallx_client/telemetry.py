"""Rerun display and timestamped rollout recording from existing observations."""

import json
import os
import queue
import threading
import time

from .contracts import JOINTS


class Telemetry:
    """Bounded background encoding; no hardware access and no disk IO on control ticks."""

    def __init__(self, cfg, path):
        if (
            cfg.display_data
            and not cfg.display_ip
            and not (os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY"))
        ):
            raise RuntimeError(
                "display_data=true needs a desktop terminal. Use --display_data=false "
                "over plain SSH, or --display_ip HOST --display_port PORT for a Rerun viewer. "
                "Rollout saving is independent of display_data."
            )
        import rerun as rr
        import rerun.blueprint as rrb

        self.cfg, self.path, self.rr = cfg, path, rr
        self.queue = queue.Queue(maxsize=60)
        self.error = None
        self.submitted = self.written = 0
        self.closing = threading.Event()
        self.stream = rr.RecordingStream("wallx-client", recording_id=path.name)
        self.viewer = None
        sinks = []
        try:
            if cfg.save_rollout:
                sinks.append(rr.FileSink(str(path / "rollout.rrd")))
            if cfg.display_data:
                if not cfg.display_ip:
                    self.stream.spawn(port=cfg.display_port, connect=False, memory_limit="2GB")
                host = cfg.display_ip or "127.0.0.1"
                sinks.append(rr.GrpcSink(f"rerun+http://{host}:{cfg.display_port}/proxy"))
            blueprint = rrb.Blueprint(
                rrb.Vertical(
                    rrb.Horizontal(
                        *[
                            rrb.Spatial2DView(name=name, origin=f"/observation/images/{name}")
                            for name in cfg.policy.camera_names
                        ]
                    ),
                    rrb.Horizontal(
                        rrb.TimeSeriesView(
                            name="Measured motor degrees", origin="/observation/state"
                        ),
                        rrb.TimeSeriesView(
                            name="Commanded motor degrees", origin="/action/physical"
                        ),
                    ),
                )
            )
            self.stream.set_sinks(*sinks, default_blueprint=blueprint)
            if cfg.display_data:
                self.viewer = {"url": f"rerun+http://{host}:{cfg.display_port}/proxy"}
                if not cfg.display_ip:
                    from .viewer import present_local_viewer

                    # spawn() only reuses the port; explicitly present that window
                    # once per episode, before any hardware or control ticks start.
                    self.viewer.update(present_local_viewer(cfg.display_port, path.name))
                else:
                    self.viewer["status"] = "remote"
        except BaseException:
            self.stream.disconnect()
            raise
        self.thread = threading.Thread(target=self._run, name="rollout-media", daemon=True)
        self.thread.start()

    def check(self):
        if self.error:
            raise OSError(f"Rollout media failed: {self.error}")

    def publish(self, observation, *, action=None, physical=None, phase="preview", query_id=None):
        self.check()
        if self.closing.is_set():
            raise OSError("Rollout media is already closing")
        # B601 observations own their RGB arrays; the worker retains these references.
        row = {
            "index": self.submitted,
            "wall_time": time.time(),
            "monotonic": time.monotonic(),
            "captured_wall": observation.captured_wall,
            "image_monotonic": observation.captured_monotonic,
            "state_monotonic": observation.state_monotonic,
            "capture_sequence": observation.capture_sequence,
            "camera_generation": observation.camera_generation,
            "camera_skew_s": observation.camera_skew_s,
            "phase": phase,
            "query_id": query_id,
            "observed_physical": observation.state.tolist(),
            "executed_recorded_command": None if action is None else list(action),
            "driver_target_physical": None if physical is None else list(physical),
        }
        try:
            self.queue.put_nowait((observation.views, row))
        except queue.Full as exc:
            raise OSError(
                "Rollout media queue full; cannot continue without complete recording"
            ) from exc
        self.submitted += 1

    def _run(self):
        writers = {}
        frames = 0
        metadata = None
        try:
            import cv2

            if self.cfg.save_rollout:
                metadata = (self.path / "rollout.jsonl").open("w", encoding="utf-8", buffering=1)
            while not self.closing.is_set() or not self.queue.empty():
                try:
                    images, row = self.queue.get(timeout=0.1)
                except queue.Empty:
                    continue
                self.stream.set_time("sample", sequence=row["index"])
                self.stream.set_time("capture_time", timestamp=row["captured_wall"])
                self.stream.set_time("control_time", timestamp=row["wall_time"])
                record_video = self.cfg.save_rollout and row["phase"] != "preview"
                row["video_index"] = frames if record_video else None
                for name, rgb in images.items():
                    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
                    if record_video:
                        if name not in writers:
                            writer = cv2.VideoWriter(
                                str(self.path / f"{name}.avi"),
                                cv2.VideoWriter_fourcc(*"MJPG"),
                                self.cfg.control.hz,
                                (rgb.shape[1], rgb.shape[0]),
                            )
                            writers[name] = writer
                            if not writer.isOpened():
                                raise OSError(f"Cannot open rollout video for {name}")
                        writers[name].write(bgr)
                    ok, jpeg = cv2.imencode(".jpg", bgr, [cv2.IMWRITE_JPEG_QUALITY, 90])
                    if not ok:
                        raise OSError(f"Cannot encode Rerun image for {name}")
                    self.stream.log(
                        f"observation/images/{name}",
                        self.rr.EncodedImage(
                            contents=jpeg.tobytes(),
                            media_type="image/jpeg",
                        ),
                    )
                for group, values in [
                    ("observation/state", row["observed_physical"]),
                    ("action/physical", row["driver_target_physical"]),
                    ("action/recorded", row["executed_recorded_command"]),
                ]:
                    if values is not None:
                        for joint, value in zip(JOINTS, values, strict=True):
                            self.stream.log(f"{group}/{joint}", self.rr.Scalars(float(value)))
                self.stream.log("episode/phase", self.rr.TextLog(row["phase"]))
                if metadata:
                    metadata.write(json.dumps(row, allow_nan=False) + "\n")
                frames += int(record_video)
                self.written += 1
        except Exception as exc:
            self.error = f"{type(exc).__name__}: {exc}"
        finally:
            for writer in writers.values():
                writer.release()
            if metadata:
                metadata.close()
            try:
                self.stream.flush(timeout_sec=5)
                self.stream.disconnect()
            except Exception as exc:
                self.error = self.error or f"Rerun finalization: {exc}"
            self.video_frames = frames

    def close(self):
        self.closing.set()
        self.thread.join(timeout=15)
        if self.thread.is_alive():
            self.error = self.error or "Media worker did not finish within 15 seconds"
        return {
            "display_data": self.cfg.display_data,
            "save_rollout": self.cfg.save_rollout,
            "submitted": self.submitted,
            "written": self.written,
            "video_frames": getattr(self, "video_frames", None),
            "error": self.error,
            "viewer": self.viewer,
        }
