import json
import time
from dataclasses import replace

import cv2
import numpy as np
import pytest

from wallx_client.cli import main
from wallx_client.config import validate_config
from wallx_client.recording import Recorder


def test_display_defaults_and_flags(capsys):
    assert main(["config"]) == 0
    cfg = json.loads(capsys.readouterr().out)
    assert cfg["display_data"] is True and cfg["save_rollout"] is True
    assert main(["--display_data=false", "config", "--save-rollout=false"]) == 0
    cfg = json.loads(capsys.readouterr().out)
    assert cfg["display_data"] is False and cfg["save_rollout"] is False
    assert (
        main(["config", "--display-data=true", "--display_ip=localhost", "--display_port=9877"])
        == 0
    )
    cfg = json.loads(capsys.readouterr().out)
    assert (cfg["display_data"], cfg["display_ip"], cfg["display_port"]) == (
        True,
        "localhost",
        9877,
    )


@pytest.mark.parametrize(
    "changes",
    [
        {"display_data": "false"},
        {"save_rollout": "true"},
        {"display_port": 0},
        {"display_port": True},
        {"display_ip": ""},
    ],
)
def test_display_validation(cfg, changes):
    with pytest.raises(ValueError):
        validate_config(replace(cfg, **changes))


def test_headless_display_fails_before_hardware(monkeypatch, tmp_path):
    monkeypatch.delenv("DISPLAY", raising=False)
    monkeypatch.delenv("WAYLAND_DISPLAY", raising=False)

    def forbidden(*args):
        raise AssertionError("Must not open hardware on display setup failure")

    monkeypatch.setattr("wallx_client.cli.hardware_session", forbidden)
    assert main(["--output", str(tmp_path / "headless"), "preflight"]) == 1
    report = json.loads((tmp_path / "headless/report.json").read_text())
    assert "desktop terminal" in report["error"]


def test_rollout_without_display_retains_video_and_timestamps(cfg, observation):
    cfg = replace(cfg, display_data=False)
    recorder = Recorder(cfg, "test")
    recorder.start_telemetry()
    recorder.sample(observation)  # Preview appears in Rerun, not the control video.
    for index in range(4):
        obs = replace(
            observation,
            capture_sequence=index,
            captured_wall=observation.captured_wall + index / 30,
        )
        recorder.sample(obs, action=[1.0] * 7, physical=[2.0] * 7, phase="policy", query_id="q1")
    recorder.close({"outcome": "fault", "error": "simulated camera disconnect"})
    report = json.loads((recorder.path / "report.json").read_text())
    assert report["outcome"] == "fault"
    assert report["rollout_media"]["written"] == 5
    assert report["rollout_media"]["video_frames"] == 4
    assert report["log_error"] is None
    assert (recorder.path / "rollout.rrd").stat().st_size > 1000
    rows = [json.loads(line) for line in (recorder.path / "rollout.jsonl").read_text().splitlines()]
    assert [row["video_index"] for row in rows] == [None, 0, 1, 2, 3]
    assert rows[-1]["driver_target_physical"] == [2.0] * 7
    assert rows[-1]["query_id"] == "q1"
    assert rows[-1]["captured_wall"] == observation.captured_wall + 3 / 30
    for name in cfg.policy.camera_names:
        video = cv2.VideoCapture(str(recorder.path / f"{name}.avi"))
        count = 0
        while True:
            ok, frame = video.read()
            if not ok:
                break
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            np.testing.assert_allclose(rgb.mean(axis=(0, 1)), [255, 30, 15], atol=5)
            count += 1
        video.release()
        assert count == 4


def test_media_write_failure_is_reported_and_control_can_detect_it(cfg, observation, monkeypatch):
    recorder = Recorder(replace(cfg, display_data=False), "test")
    recorder.start_telemetry()

    class FailedWriter:
        def __init__(self, *args):
            pass

        def isOpened(self):
            return False

        def release(self):
            pass

    monkeypatch.setattr(cv2, "VideoWriter", FailedWriter)
    recorder.sample(observation, phase="hold")
    deadline = time.monotonic() + 5
    while recorder.telemetry.error is None and time.monotonic() < deadline:
        time.sleep(0.01)
    with pytest.raises(OSError, match="Cannot open rollout video"):
        recorder.check_media()
    with pytest.raises(OSError):
        recorder.close({"outcome": "fault"})
    assert json.loads((recorder.path / "report.json").read_text())["log_error"]


def test_disabled_media_preserves_command_logs(cfg, observation):
    recorder = Recorder(replace(cfg, display_data=False, save_rollout=False), "test")
    recorder.start_telemetry()
    assert recorder.telemetry is None
    recorder.sample(observation)
    recorder.append("command", phase="hold")
    recorder.close({"outcome": "aborted"})
    assert (recorder.path / "commands.jsonl").stat().st_size > 0
    assert not (recorder.path / "rollout.rrd").exists()


@pytest.mark.parametrize("display_ip", [None, "viewer.example"])
def test_new_episode_presents_only_its_local_viewer(cfg, monkeypatch, display_ip):
    import rerun as rr

    monkeypatch.setenv("DISPLAY", ":test")
    presented, spawned, sinks = [], [], []

    class Stream:
        def __init__(self, *args, **kwargs):
            pass

        def spawn(self, **kwargs):
            spawned.append(kwargs)

        def set_sinks(self, *values, **kwargs):
            sinks.append(values)

        def flush(self, **kwargs):
            pass

        def disconnect(self):
            pass

    def present(port, recording):
        presented.append((port, recording))
        return {"status": "activated"}

    monkeypatch.setattr(rr, "RecordingStream", Stream)
    monkeypatch.setattr("wallx_client.viewer.present_local_viewer", present)
    cfg = replace(cfg, display_data=True, display_ip=display_ip, display_port=19877)
    runs = []
    for _ in range(2):
        recorder = Recorder(cfg, "test")
        recorder.start_telemetry()
        runs.append(recorder.path.name)
        recorder.close({"outcome": "complete"})
        report = json.loads((recorder.path / "report.json").read_text())
        assert report["rollout_media"]["viewer"]["status"] == (
            "remote" if display_ip else "activated"
        )
    assert presented == ([] if display_ip else [(19877, name) for name in runs])
    assert len(spawned) == (0 if display_ip else 2)
    assert all([type(s).__name__ for s in values] == ["FileSink", "GrpcSink"] for values in sinks)
