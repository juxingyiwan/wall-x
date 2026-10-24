import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest

from wallx_client import viewer


@pytest.mark.parametrize("args", [["--port=9877"], ["--port", "9877"]])
def test_viewer_selection_excludes_other_ports_and_non_viewer_processes(tmp_path, args):
    proc = tmp_path / "123"
    proc.mkdir()
    (proc / "exe").symlink_to(tmp_path / "rerun")
    (proc / "cmdline").write_bytes("\0".join(["/sdk/rerun", *args, ""]).encode())
    assert viewer._is_viewer(123, 9877, proc=tmp_path)
    assert not viewer._is_viewer(123, 9878, proc=tmp_path)
    (proc / "exe").unlink()
    (proc / "exe").symlink_to(tmp_path / "python")
    assert not viewer._is_viewer(123, 9877, proc=tmp_path)
    assert not viewer._is_viewer(456, 9877, proc=tmp_path)


def test_presentation_runs_once_with_a_deadline(monkeypatch):
    monkeypatch.setenv("DISPLAY", ":test")
    monkeypatch.setattr(viewer.sys, "platform", "linux")
    calls = []

    def run(command, **kwargs):
        calls.append((command, kwargs))
        return SimpleNamespace(stdout='{"status": "activated", "pid": 123}')

    monkeypatch.setattr(viewer.subprocess, "run", run)
    assert viewer.present_local_viewer(9877, "episode-2")["status"] == "activated"
    command, options = calls.pop()
    assert command == [
        viewer.sys.executable,
        str(Path(viewer.__file__).resolve()),
        "9877",
        "episode-2",
    ]
    assert options["timeout"] == 3
    assert not calls


def test_unresponsive_desktop_does_not_fail_recording(monkeypatch, caplog):
    monkeypatch.setenv("DISPLAY", ":test")
    monkeypatch.setattr(viewer.sys, "platform", "linux")

    def timeout(command, **kwargs):
        raise subprocess.TimeoutExpired(command, kwargs["timeout"])

    monkeypatch.setattr(viewer.subprocess, "run", timeout)
    assert viewer.present_local_viewer(9877, "episode")["status"] == "activation_failed"
    assert "could not be activated" in caplog.text


def test_headless_does_not_open_a_desktop(monkeypatch):
    monkeypatch.delenv("DISPLAY", raising=False)

    def forbidden(*args, **kwargs):
        raise AssertionError("Headless clients must not open a desktop")

    monkeypatch.setattr(viewer.subprocess, "run", forbidden)
    assert viewer.present_local_viewer(9877, "episode")["status"] == "unsupported_desktop"
