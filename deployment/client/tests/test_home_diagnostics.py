"""Regression checks use fake motors only and never open CAN or cameras."""

import logging
from contextlib import nullcontext
from dataclasses import replace
from types import SimpleNamespace

import pytest
from test_shutdown import fake_arm

from wallx_client.hardware.homing import HomingTimeout


@pytest.mark.parametrize(
    "position, stage, joint",
    [
        ([0, 0, 0, 0, 0, 0, 3.52694], "verify_zero", "gripper"),
        ([0, 5, 0, 0, 0, 0, 0], "native_arm", "shoulder_lift"),
    ],
)
def test_timeout_keeps_last_feedback_and_never_retries_home(cfg, position, stage, joint):
    events = []
    hw, history = fake_arm(
        cfg, position=position, event=lambda kind, **row: events.append((kind, row))
    )
    hw.shutdown_cfg = replace(hw.shutdown_cfg, home_timeout_s=0.2)
    hw._send = lambda target: None  # Feedback stays fixed despite commanded motion.
    with pytest.raises(HomingTimeout) as error:
        hw._home(hw.control_cfg)
    context = error.value.homing_context
    assert context["stage"] == stage
    assert context["last_feedback"]["outside_tolerance"] == [joint]
    assert context["last_feedback"]["measured_physical_deg"] == position
    assert any(kind == "home_progress" for kind, _ in events)
    assert events[-1] == ("home_failed", context)
    assert hw._home_progress_state is None
    with pytest.raises(Exception, match="Homing inhibited"):
        hw.close()
    assert history.count("home") == 1
    assert sum(value.startswith("disable:") for value in history) == 7


def test_home_diagnostics_cannot_break_successful_homing(cfg):
    def failed_recorder(*args, **kwargs):
        raise OSError("disk full")

    hw, history = fake_arm(cfg, event=failed_recorder)
    measured = hw._home(hw.control_cfg)
    assert measured == [0] * 7
    assert hw._home_progress_state is None
    assert hw._home_inhibited is None


@pytest.mark.parametrize(
    "arm_deg, gripper_deg, accepted",
    [
        (0.879, 0.4, True),
        (0.879, 2.483, False),
        (0.879, 2.461, False),
        (0, 0.5, True),
        (0, 3.001, False),
        (2.001, 2.483, False),
    ],
)
def test_zero_refinement_rejects_residual_despite_legacy_tolerance(
    cfg, arm_deg, gripper_deg, accepted
):
    hw, _ = fake_arm(cfg, position=[arm_deg, 0, 0, 0, 0, 0, gripper_deg])
    assert cfg.control.settle_gripper_deg == 3.0
    hw.shutdown_cfg = replace(hw.shutdown_cfg, home_timeout_s=0.2)
    hw._send = lambda target: None  # Simulate the observed closed mechanical stop.
    if accepted:
        result = hw._home(hw.control_cfg)
        assert result[6] == gripper_deg
        assert hw._last_home_feedback["outside_tolerance"] == []
    else:
        with pytest.raises(HomingTimeout):
            hw._home(hw.control_cfg)


def test_session_reports_homing_timeout_and_retains_cleanup(cfg, tmp_path, monkeypatch, caplog):
    from wallx_client import cli as local
    from wallx_client import controller
    from wallx_client.hardware import b601 as b601_hardware

    closed = []
    logger = logging.getLogger("wallx_client.hardware.b601")
    monkeypatch.setattr(logger, "level", logging.WARNING)
    context = {"stage": "verify_zero", "last_feedback": {"gripper_error_deg": 2.53}}

    def enable(_):
        logger.info("Homing diagnostic visibility test")
        error = HomingTimeout("Native homing exceeded 65s")
        error.homing_context = context
        raise error

    hw = SimpleNamespace(
        robot=SimpleNamespace(),
        open_observation=lambda: None,
        manifest=lambda: {},
        enable=enable,
        close=lambda: closed.append(True),
        shutdown_report={"homing": "skipped", "disable": "commands_sent"},
    )
    monkeypatch.setattr(b601_hardware, "B601", lambda *a, **k: hw)
    monkeypatch.setattr(controller, "preflight", lambda *a: None)
    monkeypatch.setattr(controller, "operator_terminal", lambda: nullcontext(lambda: None))
    monkeypatch.setattr(controller, "control_loop", lambda *a: pytest.fail("policy ran"))
    cfg = replace(cfg, robot=replace(cfg.robot, camera_mapping_verified=True))
    recorder = SimpleNamespace(
        path=tmp_path, event=lambda *a, **k: None, start_telemetry=lambda: None
    )
    monkeypatch.setattr(local, "require_execution_ready", lambda _: None)
    monkeypatch.setattr(local.sys.stdin, "isatty", lambda: True)
    cfg = replace(cfg, robot=replace(cfg.robot, lock_paths=(str(tmp_path / "robot.lock"),)))
    report = local.hardware_session(cfg, "run", recorder, None)
    assert report["outcome"] == "fault"
    assert report["policy_commands"] == 0
    assert report["error"] == "HomingTimeout: Native homing exceeded 65s"
    assert report["homing_diagnostic"] == context
    assert report["shutdown"]["disable"] == "commands_sent"
    assert closed == [True]
    assert "Homing diagnostic visibility test" in caplog.text
