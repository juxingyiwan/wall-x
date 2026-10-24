import json
import os
import subprocess
import sys
from dataclasses import replace
from pathlib import Path

import pytest
import yaml

from wallx_client.config import (
    load_config,
    require_execution_ready,
)
from wallx_client.contracts import HardwareError
from wallx_client.locks import robot_locks
from wallx_client.recording import Recorder


def test_unknown_config_key_rejected(cfg, tmp_path):
    data = cfg.to_dict()
    data["control"]["executon_horizon"] = 8
    path = tmp_path / "bad.yml"
    path.write_text(yaml.safe_dump(data))
    with pytest.raises(ValueError, match="Unknown"):
        load_config(path)


def test_execution_requires_verified_camera_and_checkpoint(cfg):
    with pytest.raises(ValueError, match="camera_mapping"):
        require_execution_ready(cfg)
    missing = replace(
        cfg,
        robot=replace(cfg.robot, camera_mapping_verified=True),
        policy=replace(cfg.policy, checkpoint_sha256=""),
    )
    with pytest.raises(ValueError, match="SHA256"):
        require_execution_ready(missing)


def test_shared_lock_refuses_duplicate_and_preserves_inode(tmp_path):
    lock = tmp_path / "control.lock"
    with robot_locks([str(lock)]):
        inode = lock.stat().st_ino
        with pytest.raises(HardwareError, match="reserved"), robot_locks([str(lock)]):
            pass
    with robot_locks([str(lock)]):
        assert lock.stat().st_ino == inode


def test_doctor_imports_no_driver_or_torch():
    root = Path(__file__).resolve().parents[1]
    code = "from wallx_client.cli import main; main(['doctor']); import sys; assert 'torch' not in sys.modules; assert 'motorbridge' not in sys.modules; assert 'lerobot_robot_seeed_b601' not in sys.modules"
    result = subprocess.run(
        [sys.executable, "-c", code],
        cwd=root,
        capture_output=True,
        text=True,
        check=True,
        env=dict(os.environ, PYTHONPATH=str(root / "src")),
    )
    assert json.loads(result.stdout)["hardware_opened"] is False


def test_recorder_never_overwrites_and_flushes_report(cfg, tmp_path):
    run = tmp_path / "episode"
    recorder = Recorder(cfg, "test", str(run))
    recorder.event("hello")
    recorder.close({"outcome": "aborted"})
    assert json.loads((run / "events.jsonl").read_text())["kind"] == "hello"
    assert json.loads((run / "report.json").read_text())["log_error"] is None
    with pytest.raises(FileExistsError):
        Recorder(cfg, "test", str(run))


def test_readable_capture_and_offline_input_are_identical(cfg, observation):
    import cv2
    import numpy as np

    recorder = Recorder(cfg, "capture_test")
    recorder.observation(observation, "observation")
    recorder.close({"outcome": "capture_complete"})
    with np.load(recorder.path / "observation.npz", allow_pickle=False) as snapshot:
        np.testing.assert_array_equal(snapshot["state"], observation.state)
        for key, rgb in observation.views.items():
            png = cv2.cvtColor(
                cv2.imread(str(recorder.path / f"observation_{key}.png")), cv2.COLOR_BGR2RGB
            )
            np.testing.assert_array_equal(png, rgb)
            np.testing.assert_array_equal(snapshot[key], rgb)


def test_default_config_is_independent_of_working_directory(tmp_path, monkeypatch, capsys):
    from wallx_client.cli import main

    monkeypatch.chdir(tmp_path)
    assert main(["config"]) == 0
    preview = json.loads(capsys.readouterr().out)
    assert preview["robot"]["camera_mapping_verified"] is False
    assert preview["policy"]["checkpoint_id"] == "example_experiment/12_16000"


def test_placeholder_digest_cannot_enable_execution(cfg):
    placeholder = replace(
        cfg,
        robot=replace(cfg.robot, camera_mapping_verified=True),
        policy=replace(cfg.policy, checkpoint_sha256="0" * 64),
    )
    with pytest.raises(ValueError, match="SHA256"):
        require_execution_ready(placeholder)


@pytest.mark.parametrize(
    "uri", ["ws://127.0.0.1:32196", "wss://inference.invalid", "local://wall-x"]
)
@pytest.mark.parametrize("rtc", [False, True])
def test_unverified_run_rejected_before_resources(cfg, tmp_path, monkeypatch, capsys, uri, rtc):
    from wallx_client import cli

    ready = replace(
        cfg,
        robot=replace(cfg.robot, camera_mapping_verified=False),
        policy=replace(cfg.policy, uri=uri),
        rtc=replace(cfg.rtc, enabled=rtc),
    )
    path = tmp_path / "client.yml"
    path.write_text(yaml.safe_dump(ready.to_dict()))

    def forbidden(*args, **kwargs):
        pytest.fail("Rejected execution must not create recording, transport or hardware")

    monkeypatch.setattr(cli, "Recorder", forbidden)
    monkeypatch.setattr(cli, "hardware_session", forbidden)
    assert cli.main(["--config", str(path), "run"], connection_factory=forbidden) == 1
    assert "camera_mapping" in capsys.readouterr().err


@pytest.mark.parametrize("rtc", [False, True])
def test_execution_gate_allows_local_and_network(cfg, rtc):
    ready = replace(
        cfg,
        robot=replace(cfg.robot, camera_mapping_verified=True),
        rtc=replace(cfg.rtc, enabled=rtc),
    )
    require_execution_ready(ready)
    require_execution_ready(
        replace(
            ready,
            policy=replace(ready.policy, uri="local://wall-x"),
        )
    )


@pytest.mark.parametrize("uri", ["ws://127.0.0.1:32196", "local://wall-x"])
@pytest.mark.parametrize("rtc", [False, True])
def test_cli_passes_rtc_to_hardware_session(cfg, tmp_path, monkeypatch, uri, rtc):
    from types import SimpleNamespace

    from wallx_client import cli

    cfg = replace(
        cfg,
        robot=replace(cfg.robot, camera_mapping_verified=True),
        policy=replace(cfg.policy, uri=uri),
    )
    path = tmp_path / "ready.yml"
    path.write_text(yaml.safe_dump(cfg.to_dict()))
    observed = []

    def session(cfg, mode, recorder, connection):
        observed.append((cfg.rtc.enabled, cfg.policy.uri, mode))
        return {"outcome": "aborted"}

    monkeypatch.setattr(cli, "hardware_session", session)
    monkeypatch.setattr(cli.Recorder, "start_telemetry", lambda self: None)
    assert (
        cli.main(
            [
                "--config",
                str(path),
                "run",
                "--rtc",
                str(rtc).lower(),
            ],
            connection_factory=lambda *a, **kw: SimpleNamespace(close=lambda: None),
        )
        == 0
    )
    assert observed == [(rtc, uri, "run")]
