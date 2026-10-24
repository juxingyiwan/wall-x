import signal
import subprocess
import sys
import threading
import time
from dataclasses import replace
from types import SimpleNamespace

import pytest

from wallx_client.config import resolve_execution, validate_config
from wallx_client.lifecycle import finish_episode
from wallx_client.workers import DaemonWorker


@pytest.mark.parametrize("mode", ["sync", "rtc_fifo"])
def test_cli_explicit_executor_resolves_after_overrides(cfg, tmp_path, capsys, mode):
    import json

    import yaml

    from wallx_client.cli import main

    value = replace(cfg, rtc=replace(cfg.rtc, enabled=True))
    path = tmp_path / "client.yml"
    path.write_text(yaml.safe_dump(value.to_dict()))
    assert (
        main(["--config", str(path), "config", "--executor", mode, "--guidance-mode", "pi_vjp"])
        == 0
    )
    result = json.loads(capsys.readouterr().out)
    assert result["execution_mode"] == mode
    assert result["rtc"]["enabled"] == (mode != "sync")
    assert result["rtc"]["guidance_mode"] == "pi_vjp"


def test_blocked_inference_shutdown_is_bounded_and_restores_signals():
    entered, release = threading.Event(), threading.Event()
    worker = DaemonWorker()

    def blocked():
        entered.set()
        release.wait(5)

    pending = worker.submit(blocked)
    assert entered.wait(1)
    before = signal.getsignal(signal.SIGTERM)
    order = []

    def close_hw():
        order.append("hardware")
        assert signal.getsignal(signal.SIGTERM) is not before

    def close_transport():
        order.append("transport")
        assert signal.getsignal(signal.SIGTERM) is before

    report = {"outcome": "aborted"}
    start = time.monotonic()
    try:
        finish_episode(
            SimpleNamespace(close=close_hw),
            SimpleNamespace(close=close_transport),
            report,
            pending,
            worker,
            timeout_s=0.15,
        )
        assert time.monotonic() - start < 0.5
        assert order == ["hardware", "transport"]
        assert report["pending_cancelled"] is False
        assert report["worker_cleanup"][0]["status"] == "running_daemon"
        assert report["finished_wall"] and report["stop_outcome"] == "aborted"
        assert signal.getsignal(signal.SIGTERM) is before
    finally:
        release.set()
        worker.close(1)


def test_running_inference_does_not_hold_interpreter_exit():
    script = """
import threading
from wallx_client.workers import DaemonWorker
entered = threading.Event()
def never():
    entered.set()
    threading.Event().wait()
w = DaemonWorker()
w.submit(never)
assert entered.wait(1)
assert w.close(0.01)["timed_out"]
print("EXIT_READY", flush=True)
"""
    value = subprocess.run(
        [sys.executable, "-c", script], capture_output=True, text=True, timeout=4
    )
    assert value.returncode == 0, value.stderr
    assert "EXIT_READY" in value.stdout


@pytest.mark.parametrize("rtc,mode", [(False, "sync"), (True, "rtc_fifo")])
def test_rtc_switch_selects_executor(cfg, rtc, mode):
    value = resolve_execution(replace(cfg, rtc=replace(cfg.rtc, enabled=rtc)))
    validate_config(value)
    assert value.execution_mode == mode


@pytest.mark.parametrize(
    "section,key",
    [
        ("control", "smooth_actions"),
        ("control", "arm_speed_deg_s"),
        ("control", "arm_accel_deg_s2"),
        ("control", "gripper_speed_deg_s"),
        ("control", "gripper_accel_deg_s2"),
        ("control", "settle_check_enabled"),
        ("rtc", "planning_timeout_s"),
        ("rtc", "min_free_steps"),
        ("rtc", "min_blend_steps"),
        ("rtc", "min_execution_steps"),
        ("rtc", "prefix_arm_tolerance_deg"),
        ("rtc", "prefix_gripper_tolerance_deg"),
    ],
)
def test_removed_executor_fields_fail_before_hardware(cfg, tmp_path, section, key):
    import yaml

    from wallx_client.config import load_config

    raw = cfg.to_dict()
    raw[section][key] = False
    path = tmp_path / "client.yml"
    path.write_text(yaml.safe_dump(raw))
    with pytest.raises(ValueError, match="Unknown"):
        load_config(path)


@pytest.mark.parametrize(
    "arguments",
    [
        ["--smooth-actions", "false"],
        ["--executor", "rtc_trajectory"],
        ["--arm-speed-deg-s", "75"],
        ["--gripper-accel-deg-s2", "720"],
    ],
)
def test_removed_cli_options_rejected(arguments):
    from wallx_client.cli import build_parser

    with pytest.raises(SystemExit) as exc:
        build_parser().parse_args(["config", *arguments])
    assert exc.value.code == 2


@pytest.mark.parametrize("steps", [1, 6, 16])
def test_fifo_retains_guidance_taper_configuration(cfg, steps):
    value = replace(cfg, rtc=replace(cfg.rtc, enabled=True, blend_steps=steps))
    validate_config(value)
