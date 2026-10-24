"""In-process model boundary, failure handling and unchanged controller inputs."""

import copy
import importlib
import json
import os
import signal
import socket
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import yaml

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "deployment/client/src")]
cli = importlib.import_module("deployment.cli")
runtime = importlib.import_module("deployment.local_runtime")
LocalPolicyConnection = importlib.import_module("deployment.local_policy").LocalPolicyConnection
config_module = importlib.import_module("wallx_client.config")
contracts = importlib.import_module("wallx_client.contracts")
protocol = importlib.import_module("wallx_client.protocol")


@pytest.fixture
def setup(tmp_path, monkeypatch):
    machine = tmp_path / "machine.yml"
    robot = cli.read_yaml(ROOT / "deployment/client/src/wallx_client/configs/b601.example.yml")[
        "robot"
    ]
    machine.write_text(
        yaml.safe_dump({"checkpoint_root": str(tmp_path), "client": {"robot": robot}})
    )
    base = cli.load_deployment(machine=machine)
    config = tmp_path / "client.yml"
    config.write_text(yaml.safe_dump(runtime.local_deployment(base)["client"]))
    cfg = config_module.load_config(config)
    now = time.monotonic()
    obs = contracts.Observation(
        np.arange(7, dtype=np.float32),
        {k: np.full((48, 64, 3), [255, 30, 15], dtype=np.uint8) for k in cfg.policy.camera_names},
        now,
        time.time(),
        now,
    )

    class Policy:
        metadata = {
            "wallx_client": {
                "protocol_version": 1,
                "state_schema": "joint_state_v1",
                "state_units": contracts.STATE_UNITS,
                "action_units": contracts.ACTION_UNITS,
                "state_dim": 7,
                "action_dim": 26,
                "action_horizon": 32,
                "camera_names": list(cfg.policy.camera_names),
                "checkpoint_id": cfg.policy.checkpoint_id,
                "checkpoint_sha256": cfg.policy.checkpoint_sha256,
                "request_id_echo": True,
                "padding": cfg.policy.padding,
            }
        }

        def __init__(self):
            self.received = []

        def infer(self, body):
            self.received.append((os.getpid(), copy.deepcopy(body)))
            return {"predict_action": np.arange(32 * 26, dtype=np.float32).reshape(32, 26)}

    policy = Policy()
    monkeypatch.setattr(cli, "verify_checkpoint", lambda _: None)
    monkeypatch.setattr(cli, "verify_server_compatibility", lambda: None)
    monkeypatch.setattr(runtime, "create_native_policy", lambda _: policy)
    args = SimpleNamespace(client_args=["check"], dry_run=False, log=tmp_path / "model.log")
    return SimpleNamespace(base=base, cfg=cfg, policy=policy, obs=obs, args=args)


def test_no_socket_or_subprocess_and_exact_jpeg_request(setup, monkeypatch):
    def forbidden(*a, **kw):
        pytest.fail("In-process inference attempted network/process creation")

    with monkeypatch.context() as inference_guard:
        inference_guard.setattr(socket, "socket", forbidden)
        inference_guard.setattr(subprocess, "Popen", forbidden)
        with LocalPolicyConnection(setup.cfg.policy, setup.policy) as connection:
            prediction = connection.predict(setup.obs, "task", generation=9)
    pid, body = setup.policy.received[0]
    assert pid == os.getpid()
    _, wire = protocol.encode_request(setup.obs, "task", setup.cfg.policy)
    expected = protocol.unpack(wire)
    expected["request_id"] = body["request_id"]
    assert body == expected
    assert prediction.generation == 9 and prediction.observation is setup.obs
    assert prediction.actions.shape == (32, 7)
    np.testing.assert_array_equal(prediction.actions, np.arange(32 * 26).reshape(32, 26)[:, :7])
    assert 0 < prediction.roundtrip_s < 2


def test_check_validates_model_without_listener_or_hardware(setup, monkeypatch, capsys):
    def forbidden(*a, **kw):
        pytest.fail("No network, subprocess or hardware permitted")

    monkeypatch.setattr(socket, "socket", forbidden)
    monkeypatch.setattr(subprocess, "Popen", forbidden)
    monkeypatch.setattr(cli, "run_client", forbidden)
    assert runtime.run_local(setup.base, setup.args) == 0
    report = json.loads(capsys.readouterr().out.splitlines()[-1])
    assert report["backend"] == "in_process" and report["pid"] == os.getpid()
    assert report["inference_port"] is None and not report["hardware_opened"]


def test_model_pins_match_cs_and_config_not_mutated(setup):
    before = copy.deepcopy(setup.base)
    effective = runtime.local_deployment(setup.base)
    effective["client"]["policy"]["uri"] = before["client"]["policy"]["uri"]
    assert effective == before == setup.base
    env = runtime.inference_environment(setup.base)
    _, _, native_env = cli.server_command(setup.base)
    assert env.pop("CUDA_VISIBLE_DEVICES") == native_env["CUDA_ID"]
    assert all(native_env[k] == v for k, v in env.items())
    assert "HOST" not in env and "PORT" not in env


@pytest.mark.parametrize("mode", ["doctor", "config"])
def test_inspection_does_not_load_model(setup, monkeypatch, mode):
    setup.args.client_args = [mode]
    monkeypatch.setattr(runtime, "create_native_policy", lambda _: pytest.fail("model loaded"))
    monkeypatch.setattr(cli, "run_client", lambda config, argv: 0)
    assert runtime.run_local(setup.base, setup.args) == 0


def test_failed_metadata_stops_before_hardware(setup, monkeypatch):
    setup.policy.metadata["wallx_client"]["checkpoint_id"] = "wrong/12_16000"
    setup.args.client_args = ["preflight"]
    monkeypatch.setattr(cli, "run_client", lambda *a, **kw: pytest.fail("hardware reached"))
    assert runtime.run_local(setup.base, setup.args) == 1


def test_client_failure_closes_backend_after_client_returns(setup, monkeypatch):
    setup.args.client_args = ["preflight"]
    seen = []

    def client(config, argv, *, connection_factory):
        assert config == runtime.local_deployment(setup.base)["client"]
        conn = connection_factory(setup.cfg.policy, rtc=setup.cfg.rtc)
        seen.append(conn)
        assert conn.open()
        raise RuntimeError("after hardware cleanup")

    monkeypatch.setattr(cli, "run_client", client)
    assert runtime.run_local(setup.base, setup.args) == 1
    with pytest.raises(contracts.TransportError, match="closed"):
        seen[0].open()


def test_run_requires_operator_before_loading_model(setup, monkeypatch):
    setup.args.client_args = ["run"]
    monkeypatch.setattr(config_module, "require_execution_ready", lambda _: None)
    monkeypatch.setattr(sys.stdin, "isatty", lambda: False)
    monkeypatch.setattr(runtime, "create_native_policy", lambda _: pytest.fail("model loaded"))
    with pytest.raises(ValueError, match="interactive"):
        runtime.run_local(setup.base, setup.args)


def test_environment_restored_after_model_failure(setup, monkeypatch):
    monkeypatch.setenv("WALLX_SMOOTH_ACTION", "original")

    def fail(_):
        assert os.environ["WALLX_SMOOTH_ACTION"] == "false"
        raise RuntimeError("load failed")

    monkeypatch.setattr(runtime, "create_native_policy", fail)
    assert runtime.run_local(setup.base, setup.args) == 1
    assert os.environ["WALLX_SMOOTH_ACTION"] == "original"


def test_existing_log_not_overwritten(setup):
    setup.args.log.write_text("previous")
    assert runtime.run_local(setup.base, setup.args) == 1
    assert setup.args.log.read_text() == "previous"


def test_timeout_discards_late_result_and_session_cannot_retry(setup, monkeypatch):
    gate = threading.Event()
    entered = threading.Event()

    def delayed(body):
        entered.set()
        gate.wait(2)
        return {"predict_action": np.zeros((32, 26))}

    monkeypatch.setattr(setup.policy, "infer", delayed)
    conn = LocalPolicyConnection(replace(setup.cfg.policy, request_timeout_s=0.02), setup.policy)
    try:
        with pytest.raises(contracts.TransportError, match="timed out"):
            conn.predict(setup.obs, "task")
        assert entered.is_set()
        with pytest.raises(contracts.TransportError, match="closed"):
            conn.predict(setup.obs, "task")
    finally:
        gate.set()
        conn.close()
        conn._worker.thread.join(timeout=2)
    assert not conn._worker.thread.is_alive()


def test_concurrent_request_rejected_and_close_discards_pending(setup, monkeypatch):
    gate, entered = threading.Event(), threading.Event()

    def delayed(body):
        entered.set()
        gate.wait(2)
        return {"predict_action": np.zeros((32, 26))}

    monkeypatch.setattr(setup.policy, "infer", delayed)
    conn = LocalPolicyConnection(setup.cfg.policy, setup.policy)
    with ThreadPoolExecutor(1) as pool:
        first = pool.submit(conn.predict, setup.obs, "task")
        assert entered.wait(1)
        with pytest.raises(contracts.ProtocolError, match="one in-flight"):
            conn.predict(setup.obs, "task")
        conn.close()
        gate.set()
        with pytest.raises(contracts.TransportError, match="discarded"):
            first.result(timeout=2)


def test_inner_model_cleanup_is_reported_after_outer_timeout(setup, monkeypatch):
    from wallx_client.lifecycle import finish_episode
    from wallx_client.workers import DaemonWorker

    gate, entered = threading.Event(), threading.Event()

    def blocked(body):
        entered.set()
        assert gate.wait(3)
        return {"predict_action": np.zeros((32, 26))}

    monkeypatch.setattr(setup.policy, "infer", blocked)
    conn = LocalPolicyConnection(replace(setup.cfg.policy, request_timeout_s=0.02), setup.policy)
    outer = DaemonWorker(thread_name_prefix="test-outer-local")
    pending = outer.submit(conn.predict, setup.obs, "task")
    try:
        assert entered.wait(1)
        with pytest.raises(contracts.TransportError, match="timed out"):
            pending.result(timeout=1)
        report = {"outcome": "fault"}
        order = []
        hw = SimpleNamespace(close=lambda: order.append("hardware"))
        start = time.monotonic()
        finish_episode(hw, conn, report, pending, outer, timeout_s=0.2)
        assert time.monotonic() - start < 0.5
        assert order == ["hardware"]
        backend = report["backend_cleanup"]
        assert backend["owner"] == "local_model"
        assert backend["status"] == "running_daemon" and backend["thread_alive"]
        assert backend["model_reference_held_by_worker"]
        assert not backend["session_reusable"] and conn.policy is None
        assert report["worker_cleanup"][1]["status"] == "joined"
        assert "backend cleanup: running_daemon" in report["cleanup_errors"]
        with pytest.raises(contracts.TransportError, match="closed"):
            conn.predict(setup.obs, "task")
    finally:
        gate.set()
        status = conn.close_with_status(1)
        outer.close(1)
    assert status["status"] == "joined"
    assert not status["thread_alive"] and not status["model_reference_held_by_worker"]
    assert not status["session_reusable"]


def test_local_cleanup_without_started_worker_is_idempotent(setup):
    conn = LocalPolicyConnection(setup.cfg.policy, setup.policy)
    for _ in range(2):
        status = conn.close_with_status(0.1)
        assert status["status"] == "not_started" and not status["timed_out"]
        assert not status["model_reference_held_by_worker"]
    with pytest.raises(contracts.TransportError, match="closed"):
        conn.open()


@pytest.mark.parametrize("kind", ["exception", "shape", "nan"])
def test_invalid_model_result_never_becomes_prediction(setup, monkeypatch, kind):
    def invalid(body):
        if kind == "exception":
            raise RuntimeError("CUDA failed")
        return {
            "predict_action": np.zeros((16, 26)) if kind == "shape" else np.full((32, 26), np.nan)
        }

    monkeypatch.setattr(setup.policy, "infer", invalid)
    with LocalPolicyConnection(setup.cfg.policy, setup.policy) as conn:
        with pytest.raises(contracts.ProtocolError):
            conn.predict(setup.obs, "task")


def test_termination_signal_restores_handler():
    before = signal.getsignal(signal.SIGTERM)
    with pytest.raises(KeyboardInterrupt), runtime.termination_signals():
        signal.raise_signal(signal.SIGTERM)
    assert signal.getsignal(signal.SIGTERM) == before


def test_rtc_bootstrap_and_guided_requests(setup, monkeypatch):
    rtc_module = importlib.import_module("wallx_client.rtc")
    rtc = config_module.RTCConfig(enabled=True)
    setup.policy.metadata["rtc"] = dict(
        version=1,
        supported=True,
        guidance="endpoint_vjp",
        guidance_modes=["lerobot", "pi_vjp"],
        action_horizon=32,
        real_action_dim=7,
        model_dt_s=1 / 30,
        prefix_units="recorded_commands",
    )
    seen = []

    def infer(body):
        req = body["rtc"]
        seen.append(req)
        ack = {
            k: req[k]
            for k in (
                "version",
                "epoch",
                "plan_id",
                "committed_steps",
                "guidance_horizon",
                "guidance_mode",
            )
        }
        ack["mode"] = "bootstrap" if req["prefix"] is None else "guided"
        return {"predict_action": np.zeros((32, 26)), "rtc": ack}

    monkeypatch.setattr(setup.policy, "infer", infer)
    with LocalPolicyConnection(setup.cfg.policy, setup.policy, rtc=rtc) as conn:
        assert conn.predict(setup.obs, "task").rtc["mode"] == "bootstrap"
        sent = rtc_module.request(rtc, 2, 3, np.ones((32, 7)), 14)
        assert (
            conn.predict(setup.obs, "task", generation=2, rtc_request=sent).rtc["mode"] == "guided"
        )
        assert seen[1] == sent


@pytest.mark.parametrize("mode", ["rtc_fifo"])
def test_local_runtime_and_client_share_explicit_mode_resolution(
    setup, tmp_path, monkeypatch, mode
):
    setup.args.client_args = ["preflight", "--executor", mode, "--guidance-mode", "pi_vjp"]
    setup.policy.metadata["rtc"] = dict(
        version=1,
        supported=True,
        guidance_modes=["lerobot", "pi_vjp"],
        action_horizon=32,
        real_action_dim=7,
        model_dt_s=1 / 30,
        prefix_units="recorded_commands",
    )
    observed = []

    def client(raw, args, connection_factory):
        from wallx_client.cli import apply_cli_overrides, build_parser

        path = tmp_path / "resolved.yml"
        path.write_text(yaml.safe_dump(raw))
        value = apply_cli_overrides(
            config_module.load_config(path, validate=False), build_parser().parse_args(args)
        )
        conn = connection_factory(value.policy, rtc=value.rtc)
        observed.append((value.execution_mode, conn.rtc.enabled, conn.rtc.guidance_mode))
        return 0

    monkeypatch.setattr(cli, "run_client", client)
    assert runtime.run_local(setup.base, setup.args) == 0
    assert observed == [(mode, True, "pi_vjp")]


@pytest.mark.parametrize("legacy_source", ["cli", "yaml"])
def test_retired_execution_rejected_before_loading_model(setup, monkeypatch, legacy_source):
    setup.args.client_args = ["run"]
    if legacy_source == "cli":
        setup.args.client_args += ["--continuous", "true"]
    else:
        setup.base["client"]["control"]["continuous"] = True
    monkeypatch.setattr(runtime, "native_policy", lambda *a: pytest.fail("Model loaded"))
    monkeypatch.setattr(cli, "run_client", lambda *a, **kw: pytest.fail("Client started"))
    expected = SystemExit if legacy_source == "cli" else ValueError
    with pytest.raises(expected):
        runtime.run_local(setup.base, setup.args)
