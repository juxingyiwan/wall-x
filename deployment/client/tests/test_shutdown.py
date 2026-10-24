"""Fault injection with fake motors only; no driver imports or CAN access."""

import logging
import os
import signal
import time
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from wallx_client.config import ShutdownConfig, validate_config
from wallx_client.contracts import JOINTS, HardwareError, Prediction, TransportError
from wallx_client.hardware.b601 import B601, Feedback, StaleFeedback
from wallx_client.hardware.temperature_guard import TemperatureFeedbackInvalid
from wallx_client.lifecycle import EpisodeBudget, EpisodeLimit, finish_episode


def fake_arm(cfg, *, position=None, home=None, event=None):
    history = []
    position = np.zeros(7) if position is None else np.asarray(position, dtype=float)
    hw = B601(
        cfg.robot,
        event or (lambda *a, **kw: None),
        shutdown=ShutdownConfig(home_timeout_s=0.12, settle_s=0.01),
    )
    hw.enabled = list(JOINTS)
    hw.all_enabled = True
    hw.control_cfg = replace(cfg.control, startup_gripper_speed_deg_s=6000)
    hw.directions = np.array([1, 1, -1, -1, -1, 1, 6])
    hw.limits = np.array([[-500, 500]] * 6 + [[0, 270]])
    hw.test_targets = []

    def send_action(action):
        target = np.array([action[j + ".pos"] for j in JOINTS]) * hw.directions
        hw.test_targets.append(target.copy())
        position[:] = target
        return dict(zip((j + ".pos" for j in JOINTS), target, strict=True))

    def native_home(**kwargs):
        history.append("home")
        if home:
            home(hw)
        logging.getLogger("lerobot_robot_seeed_b601.seeed_b601_follower").info("safe_zero done.")

    hw.robot = SimpleNamespace(
        safe_zero=native_home,
        send_action=send_action,
        _emergency_disable_requested=False,
        motors={
            name: SimpleNamespace(
                disable=lambda name=name: history.append("disable:" + name),
                close=lambda: history.append("motor.close"),
            )
            for name in JOINTS
        },
        bus=SimpleNamespace(
            close_bus=lambda: history.append("bus.stop"),
            close=lambda: history.append("bus.close"),
        ),
    )
    hw.feedback = SimpleNamespace(
        snapshot=lambda: (position.copy(), {}),
        refresh=lambda: (position.copy(), {}),
        close=lambda: history.append("feedback.close"),
    )
    hw.cameras = SimpleNamespace(close=lambda: history.append("cameras.close"))
    return hw, history


def test_home_once_verified_before_disable_even_when_recorder_fails(cfg):
    def log_failure(*a, **kw):
        raise OSError("disk full")

    hw, history = fake_arm(cfg, event=log_failure, position=[0, 0, 0, 0, 0, 0, 170])
    result = hw.close()
    hw.close()
    assert history.count("home") == 1
    assert history[1:8] == ["disable:" + name for name in reversed(JOINTS)]
    assert history.index("disable:shoulder_pan") < history.index("cameras.close")
    assert result["homing"] == "completed"
    assert result["measured_physical_deg"] == [0] * 7
    assert hw.test_targets[-1][6] == 0
    assert result["disable"] == "commands_sent"
    with pytest.raises(HardwareError, match="after shutdown"):
        hw.send(np.zeros(7))


def test_manifest_fingerprints_native_home_source_through_adapter_subclass(cfg):
    hw, _ = fake_arm(cfg)
    hw.directions = np.ones(7)
    hw.limits = np.array([[-500, 500]] * 7)
    assert Path(hw.manifest()["driver_source"]) == Path(__file__)


def test_startup_waits_for_gripper_arrival_even_when_arm_is_already_zero(cfg):
    hw, _ = fake_arm(cfg)
    hw.shutdown_cfg = replace(hw.shutdown_cfg, home_timeout_s=0.5)
    positions = iter([8.9, 4.0, 1.5, 1.0, 0.5])
    measured_gripper = []
    holds = []

    def feedback():
        grip = next(positions, 0.5)
        measured_gripper.append(grip)
        return np.array([0, 0, 0, 0, 0, 0, grip]), {}

    hw.feedback.refresh = feedback
    hw._send = lambda target: holds.append(target)
    measured = hw._wait_home(cfg.control)
    assert measured_gripper[:3] == [8.9, 4.0, 1.5]
    assert len(measured_gripper) >= 4  # Require successive settled readings.
    assert measured[6] <= cfg.control.settle_gripper_deg
    assert holds and all(np.array_equal(target, np.zeros(7)) for target in holds)


def test_startup_arrival_timeout_inhibits_more_homing_and_disables(cfg):
    hw, history = fake_arm(cfg, position=[0, 0, 0, 0, 0, 0, 8.9])
    hw.shutdown_cfg = replace(hw.shutdown_cfg, home_timeout_s=0.01)
    hw._send = lambda target: None
    with pytest.raises(HardwareError, match="Startup zero not reached"):
        hw._wait_home(cfg.control)
    with pytest.raises(HardwareError, match="Homing inhibited"):
        hw.close()
    assert "home" not in history
    assert hw.shutdown_report["disable"] == "commands_sent"


def test_shutdown_gripper_uses_configured_speed_and_all_joint_verification(cfg):
    events = []
    hw, history = fake_arm(
        cfg, position=[0, 0, 0, 0, 0, 0, 4], event=lambda kind, **kw: events.append(kind)
    )
    hw.control_cfg = cfg.control
    hw.shutdown_cfg = replace(hw.shutdown_cfg, home_timeout_s=0.5)
    result = hw.close()
    targets = np.array(hw.test_targets)
    assert np.max(np.abs(np.diff(np.r_[4, targets[:, 6]]))) <= 60 / 30
    assert np.all(targets[:, :6] == 0)
    assert result["measured_physical_deg"] == [0] * 7
    assert events.count("shutdown_home_started") == 1
    assert events.count("shutdown_home_complete") == 1
    assert history.count("home") == 1


@pytest.mark.parametrize("fault", ["stuck_gripper", "lost_feedback"])
def test_shutdown_gripper_failure_is_not_success_and_still_disables(cfg, fault):
    hw, history = fake_arm(cfg, position=[0, 0, 0, 0, 0, 0, 8.9])
    sent = []

    def send(target):
        sent.append(target.copy())  # Motor does not follow the target.
        if fault == "lost_feedback":

            def lost():
                raise HardwareError("CAN lost during gripper home")

            hw.feedback.refresh = lost

    hw._send = send
    with pytest.raises(HardwareError):
        hw.close()
    assert sent
    if fault == "lost_feedback":
        assert len(sent) == 1
    assert hw.shutdown_report["homing"] == "failed"
    assert hw.shutdown_report["disable"] == "commands_sent"
    assert "measured_physical_deg" not in hw.shutdown_report
    assert sum(x.startswith("disable:") for x in history) == 7


@pytest.mark.parametrize("fault", ["stale", "emergency", "tracking", "partial"])
def test_unsafe_or_partial_session_never_homes_or_reenables(cfg, fault):
    hw, history = fake_arm(cfg)
    if fault == "stale":

        def stale():
            raise HardwareError("Stale CAN feedback")

        hw.feedback.snapshot = stale
    elif fault == "emergency":
        hw.robot._emergency_disable_requested = True
    elif fault == "tracking":
        hw.inhibit_home("possible obstruction")
    else:
        hw.all_enabled = False
    if fault == "partial":
        hw.close()
    else:
        with pytest.raises(HardwareError, match="Homing inhibited"):
            hw.close()
        assert hw.shutdown_report["homing"] == "skipped"
    assert "home" not in history
    assert sum(x.startswith("disable:") for x in history) == 7


@pytest.mark.parametrize("fault", ["wrong_position", "blocked_driver", "driver_error", "feedback"])
def test_homing_failure_always_disables_and_is_not_reported_as_success(cfg, fault):
    def home(hw):
        if fault == "blocked_driver":
            time.sleep(10)  # Interrupted by the actual process homing deadline.
        elif fault == "driver_error":
            logging.getLogger("lerobot_robot_seeed_b601.seeed_b601_follower").error("home failed")
        elif fault == "feedback":

            def lost():
                raise HardwareError("lost feedback during home")

            hw.feedback.refresh = lost

    hw, history = fake_arm(
        cfg, position=np.ones(7) * (20 if fault == "wrong_position" else 0), home=home
    )
    with pytest.raises(HardwareError):
        hw.close()
    assert hw.shutdown_report["homing"] == "failed"
    assert hw.shutdown_report["disable"] == "commands_sent"
    assert history.count("home") == 1
    hw.close()
    assert history.count("home") == 1


def test_second_ctrl_c_during_homing_does_not_interrupt_disable_and_handlers_restore(cfg):
    original = signal.getsignal(signal.SIGINT)
    hw, history = fake_arm(cfg, home=lambda _: os.kill(os.getpid(), signal.SIGINT))
    hw.close()
    assert hw.shutdown_report["homing"] == "completed"
    assert len([x for x in history if x.startswith("disable:")]) == 7
    assert signal.getsignal(signal.SIGINT) == original


def test_one_disable_failure_does_not_skip_remaining_motors(cfg):
    hw, history = fake_arm(cfg)

    def fail():
        raise OSError("CAN send failed")

    hw.robot.motors[JOINTS[-1]].disable = fail
    with pytest.raises(HardwareError, match="CAN send failed"):
        hw.close()
    assert len([x for x in history if x.startswith("disable:")]) == 6
    assert history[-1] == "bus.close"
    assert hw.shutdown_report["disable"] == "failed"


def test_feedback_fault_latches_before_another_refresh(monkeypatch):
    fb = Feedback.__new__(Feedback)
    fb.error = None

    def bad():
        raise HardwareError("motor temperature")

    monkeypatch.setattr(fb, "_snapshot", bad)
    with pytest.raises(HardwareError, match="temperature"):
        fb.snapshot()
    with pytest.raises(HardwareError, match="temperature"):
        fb.refresh()  # Must fail before touching a motor, even if it recovered.


def test_native_action_guard_checks_feedback_before_driver_dispatch(cfg, monkeypatch):
    import sys

    import wallx_client.hardware.b601 as module

    sent = []

    class Driver:
        def __init__(self, config):
            self.config = config
            self.motors = {}

        def _add_motors_to_bus(self):
            pass

        def send_action(self, action):
            sent.append(action)
            return action

    def config(**kwargs):
        return SimpleNamespace(
            max_relative_target=None,
            joint_directions=dict.fromkeys(JOINTS, 1),
            joint_limits=dict.fromkeys(JOINTS, (-500, 500)),
            **kwargs,
        )

    feedback = SimpleNamespace(snapshot=lambda: (np.zeros(7), {}), refresh=lambda: None)
    monkeypatch.setitem(
        sys.modules,
        "lerobot_robot_seeed_b601",
        SimpleNamespace(
            SeeedB601RSFollower=Driver,
            SeeedB601RSFollowerConfig=config,
        ),
    )
    monkeypatch.setitem(
        sys.modules, "motorbridge", SimpleNamespace(Controller=lambda **k: object())
    )
    monkeypatch.setattr(module, "Feedback", lambda *a: feedback)
    monkeypatch.setattr(
        module,
        "Cameras",
        lambda *a, **kw: SimpleNamespace(
            latest=True,
            error=None,
            snapshot=lambda: None,
        ),
    )
    hw = B601(cfg.robot, lambda *a, **k: None)
    hw.open_observation()
    hw._home_active = True
    hw.robot.send_action({"shoulder_pan.pos": 0})

    def lost():
        raise HardwareError("CAN lost during native interpolation")

    feedback.refresh = lost
    with pytest.raises(HardwareError, match="CAN lost"):
        hw.robot.send_action({"shoulder_pan.pos": 1})
    # A later successful poll cannot clear an already latched homing fault.
    feedback.refresh = lambda: (np.zeros(7), {})
    with pytest.raises(HardwareError, match="CAN lost"):
        hw.robot.send_action({"shoulder_pan.pos": 2})
    assert sent == [{"shoulder_pan.pos": 0}]


@pytest.mark.parametrize("interrupted", [False, True])
def test_partial_startup_failure_has_shutdown_report(cfg, tmp_path, monkeypatch, interrupted):
    from contextlib import contextmanager

    from wallx_client.cli import hardware_session

    hw, history = fake_arm(cfg)
    hw.all_enabled = False
    hw.open_observation = lambda: None
    hw.manifest = lambda: {}

    def enable(*_):
        if interrupted:
            raise KeyboardInterrupt
        raise HardwareError("partial enable failed")

    hw.enable = enable

    @contextmanager
    def terminal():
        yield lambda: ""

    monkeypatch.setattr("wallx_client.hardware.b601.B601", lambda *a, **k: hw)
    monkeypatch.setattr("wallx_client.controller.operator_terminal", terminal)
    monkeypatch.setattr("wallx_client.controller.preflight", lambda *a: None)
    monkeypatch.setattr("wallx_client.cli.sys.stdin.isatty", lambda: True)
    cfg = replace(
        cfg,
        robot=replace(
            cfg.robot, camera_mapping_verified=True, lock_paths=(str(tmp_path / "robot.lock"),)
        ),
    )
    recorder = SimpleNamespace(path=tmp_path, event=lambda *a, **k: None)
    report = hardware_session(cfg, "run", recorder, object())
    assert report["outcome"] == ("aborted" if interrupted else "fault")
    assert report["shutdown"]["homing"] == "not_required"
    assert "home" not in history
    assert sum(x.startswith("disable:") for x in history) == 7


def test_homing_does_not_replace_an_existing_alarm(cfg):
    original = signal.getsignal(signal.SIGALRM)
    signal.setitimer(signal.ITIMER_REAL, 10)
    try:
        hw, history = fake_arm(cfg)
        with pytest.raises(HardwareError, match="active process alarm"):
            hw.close()
        assert "home" not in history
        assert signal.getitimer(signal.ITIMER_REAL)[0] > 0
        assert signal.getsignal(signal.SIGALRM) == original
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)


def test_execution_budget_includes_pauses_and_never_means_task_success(cfg):
    budget = EpisodeBudget(replace(cfg.control, duration_s=3, max_policy_chunks=1), started=10)
    budget.plan_started()
    with pytest.raises(EpisodeLimit, match="max_policy_chunks"):
        budget.plan_started()
    assert budget.chunks == 1
    with pytest.raises(EpisodeLimit, match="duration_s"):
        budget.check_time(13)


@pytest.mark.parametrize("name", ["max_policy_chunks"])
@pytest.mark.parametrize("value", [0, -1, True, 1.5])
def test_invalid_episode_limit_rejected(cfg, name, value):
    with pytest.raises(ValueError, match=name):
        validate_config(replace(cfg, control=replace(cfg.control, **{name: value})))


@pytest.mark.parametrize(
    "name", ["home_timeout_s", "arm_tolerance_deg", "settle_s", "feedback_recovery_timeout_s"]
)
@pytest.mark.parametrize("value", [0, -1, True, float("nan"), float("inf")])
def test_invalid_homing_limit_rejected(cfg, name, value):
    with pytest.raises(ValueError, match=name):
        validate_config(replace(cfg, shutdown=replace(cfg.shutdown, **{name: value})))


@pytest.mark.parametrize("value", [0.25, 6])
def test_recovery_budget_must_allow_settling_but_remain_bounded(cfg, value):
    with pytest.raises(ValueError, match="feedback_recovery_timeout_s"):
        validate_config(
            replace(cfg, shutdown=replace(cfg.shutdown, feedback_recovery_timeout_s=value))
        )


@pytest.mark.parametrize("fault_class", [StaleFeedback, TemperatureFeedbackInvalid])
def test_feedback_exit_recovers_then_homes_before_disable_preserving_last_target(cfg, fault_class):
    hw, history = fake_arm(cfg)
    hw.last_target = np.array([1, 2, 3, 4, 5, 6, 7.0])
    healthy = hw.feedback.snapshot

    def stale():
        raise fault_class("invalid feedback")

    def recover(timeout, settle, hold):
        history.append("recovery")
        assert hw._closed and hw.all_enabled
        hold(np.zeros(7))
        np.testing.assert_array_equal(hw.test_targets[-1], [1, 2, 3, 4, 5, 6, 7])
        hw.feedback.snapshot = healthy
        # Simulate native arm arrival; the real native path is tested separately.
        hw.robot.safe_zero = lambda **kw: (
            history.append("home"),
            hw._send(np.zeros(7)),
            logging.getLogger("lerobot_robot_seeed_b601.seeed_b601_follower").info(
                "safe_zero done."
            ),
        )

    hw.feedback.snapshot = stale
    hw.feedback.recover_for_shutdown = recover
    result = hw.close()
    assert history[:2] == ["recovery", "home"]
    assert result["feedback_recovery"] == "recovered"
    assert result["feedback_fault_type"] == fault_class.__name__
    assert result["homing"] == "completed"
    assert result["disable_reason"] == "home_verified"
    assert history.count("home") == 1
    hw.close()
    assert history.count("recovery") == 1


@pytest.mark.parametrize("fault", ["timeout", "blocked_io", "emergency", "send", "new_stale"])
def test_failed_recovery_records_why_home_is_impossible_before_disabling(cfg, fault):
    hw, history = fake_arm(cfg)
    hw.shutdown_cfg = replace(hw.shutdown_cfg, feedback_recovery_timeout_s=0.03)

    def stale():
        raise StaleFeedback("Stale CAN feedback")

    def recover(timeout, settle, hold):
        history.append("recovery")
        if fault == "blocked_io":
            time.sleep(10)
        if fault == "emergency":
            hw.robot._emergency_disable_requested = True
            hold(np.zeros(7))
        if fault == "send":
            hw.inhibit_home("Motor command failed")
            hold(np.zeros(7))
        if fault == "new_stale":
            return  # The final check must still reject stale feedback.
        raise HardwareError("Recovery timed out")

    hw.feedback.snapshot = stale
    hw.feedback.recover_for_shutdown = recover
    with pytest.raises(HardwareError):
        hw.close()
    assert history[0] == "recovery" and "home" not in history
    assert len([v for v in history if v.startswith("disable:")]) == 7
    assert hw.shutdown_report["feedback_recovery"] == "failed"
    assert hw.shutdown_report["disable_reason"] == "home_unavailable"
    assert hw.shutdown_report["feedback_recovery_error"]


@pytest.mark.parametrize("fault_class", [StaleFeedback, TemperatureFeedbackInvalid])
def test_new_fault_during_home_does_not_restart_a_partly_executed_trajectory(cfg, fault_class):
    def home(_):
        raise fault_class("invalid feedback during home")

    hw, history = fake_arm(cfg, home=home)
    with pytest.raises(HardwareError, match="invalid feedback during home"):
        hw.close()
    assert history.count("home") == 1
    assert hw.shutdown_report["disable_reason"] == "home_failed"
    assert len([v for v in history if v.startswith("disable:")]) == 7


def test_failed_startup_home_never_replays_an_old_policy_hold_during_cleanup(cfg):
    def home(_):
        raise StaleFeedback("CAN stale during startup home")

    hw, history = fake_arm(cfg, home=home)
    with pytest.raises(StaleFeedback):
        hw._home(cfg.control)
    hw.feedback.recover_for_shutdown = lambda *args: pytest.fail("restarted a failed home")
    with pytest.raises(HardwareError, match="Homing inhibited"):
        hw.close()
    assert history.count("home") == 1
    assert hw.shutdown_report["disable_reason"] == "home_unavailable"


def test_successful_shutdown_recovery_preserves_original_episode_failure(cfg):
    hw, _ = fake_arm(cfg)
    good = hw.feedback.snapshot

    def stale():
        raise StaleFeedback("Stale CAN feedback")

    def recover(*_):
        hw.feedback.snapshot = good

    hw.feedback.snapshot = stale
    hw.feedback.recover_for_shutdown = recover
    report = {"outcome": "fault", "error": "Stale CAN feedback"}
    finish_episode(hw, SimpleNamespace(close=lambda: None), report, None)
    assert report["outcome"] == report["stop_outcome"] == "fault"
    assert report["error"] == "Stale CAN feedback"
    assert report["shutdown"]["homing"] == "completed"
    assert report["shutdown"]["feedback_recovery"] == "recovered"


@pytest.mark.parametrize("scenario", ["recovers", "persistent", "fails_during_home"])
def test_temperature_fault_full_shutdown_lifecycle_with_real_feedback_guard(
    cfg, monkeypatch, scenario
):
    from test_feedback_recovery import feedback_rig, latch_temperature_fault

    fb, clock, _, states = feedback_rig(cfg, monkeypatch)
    for state in states.values():
        state.pos = 0
    latch_temperature_fault(fb, clock, states)
    original_error = str(fb.error)
    started = clock[0]
    report_motor2 = fb.robot.motors[JOINTS[1]].robstride_set_active_report

    def report(_):
        if scenario != "persistent" and clock[0] - started >= 0.1:
            states[JOINTS[1]].t_mos = 35
        report_motor2(True)

    fb.robot.motors[JOINTS[1]].robstride_set_active_report = report

    def home(_):
        if scenario == "fails_during_home":
            states[JOINTS[1]].t_mos = -20
            fb.robot.motors[JOINTS[1]].robstride_set_active_report = report_motor2

    hw, history = fake_arm(cfg, home=home)
    hw.shutdown_cfg = replace(hw.shutdown_cfg, settle_s=0.25, home_timeout_s=1)
    hw.feedback = fb
    fb.close = lambda: history.append("feedback.close")
    result = {"outcome": "fault", "error": original_error}
    finish_episode(hw, SimpleNamespace(close=lambda: None), result, None)
    assert result["stop_outcome"] == result["outcome"] == "fault"
    assert result["error"] == original_error
    assert sum(item.startswith("disable:") for item in history) == 7
    if scenario == "persistent":
        assert history.count("home") == 0 and hw.test_targets == []
        assert result["shutdown"]["feedback_recovery"] == "failed"
        assert result["shutdown"]["disable_reason"] == "home_unavailable"
    else:
        assert history.count("home") == 1
        assert history.index("home") < history.index("disable:gripper")
        assert result["shutdown"]["feedback_recovery"] == "recovered"
        assert result["shutdown"]["disable_reason"] == (
            "home_verified" if scenario == "recovers" else "home_failed"
        )
    with pytest.raises(HardwareError, match="after shutdown"):
        hw.send(np.zeros(7))
    hw.close()
    assert sum(item.startswith("disable:") for item in history) == 7


def test_pending_temperature_at_exit_also_requires_shutdown_revalidation(cfg):
    hw, history = fake_arm(cfg)
    healthy = hw.feedback.snapshot
    hw.feedback.snapshot = lambda: (
        np.zeros(7),
        {"temperature_feedback_pending": {"shoulder_lift": {}}},
    )

    def recover(*_):
        history.append("recovery")
        hw.feedback.snapshot = healthy

    hw.feedback.recover_for_shutdown = recover
    result = hw.close()
    assert history[:2] == ["recovery", "home"]
    assert result["feedback_recovery"] == "recovered"


def test_cleanup_failure_preserves_original_stop_reason_and_closes_every_worker(cfg):
    order = []
    hw, _ = fake_arm(cfg)

    def fail():
        order.append("network")
        raise OSError("network close failed")

    pool = SimpleNamespace(shutdown=lambda **kw: order.append("worker"))
    report = {"outcome": "aborted"}
    finish_episode(hw, SimpleNamespace(close=fail), report, None, pool, pool)
    assert order == ["network", "worker", "worker"]
    assert report["stop_outcome"] == "aborted"
    assert report["outcome"] == "fault"
    assert report["shutdown"]["homing"] == "completed"


@pytest.mark.parametrize("rtc", [False, True])
@pytest.mark.parametrize("fault", ["network", "camera", "interrupt", "time", "chunks"])
def test_both_controllers_terminate_and_close_once(cfg, observation, tmp_path, rtc, fault):
    from wallx_client.controller import control_loop as serial
    from wallx_client.rtc_queue_controller import control_loop as asynchronous

    cfg = replace(
        cfg,
        control=replace(
            cfg.control,
            duration_s=0.1 if fault == "time" else 4,
            max_policy_chunks=1,
        ),
    )
    calls = []
    closed = []
    sent = []

    class Hardware:
        directions = np.ones(7)
        limits = np.array([[-500, 500]] * 7)

        def observe(self):
            if fault == "camera":
                raise HardwareError("camera disconnected")
            now = time.monotonic()
            return replace(
                observation, state=np.zeros(7), captured_monotonic=now, state_monotonic=now
            ), {}

        def hold_command(self, state):
            return state

        def send(self, raw):
            assert "hardware" not in closed
            sent.append(raw.copy())
            return {"driver_target_physical": raw.tolist()}

        def close(self):
            closed.append("hardware")

    class Connection:
        def predict(self, obs, instruction, generation, **kwargs):
            calls.append(generation)
            if fault == "network":
                raise TransportError("server disconnected")
            return Prediction(
                "x",
                generation,
                np.zeros((32, 7)),
                obs,
                time.monotonic(),
                0.001,
                {},
                {"mode": "bootstrap"},
            )

        def close(self):
            closed.append("network")

    recorder = SimpleNamespace(
        path=tmp_path,
        event=lambda *a, **k: None,
        prediction=lambda *a: None,
        prediction_async=lambda *a: None,
        check_media=lambda: None,
        sample=lambda *a, **k: None,
        append=lambda *a, **k: None,
    )

    def key():
        if fault == "interrupt":
            raise KeyboardInterrupt
        return ""

    report = (asynchronous if rtc else serial)(Hardware(), Connection(), cfg, recorder, key)
    assert closed == ["hardware", "network"]
    assert (
        report["outcome"]
        == {
            "network": "fault",
            "camera": "fault",
            "interrupt": "aborted",
            "time": "timeout",
            "chunks": "budget_exhausted",
        }[fault]
    ), report
    if fault == "chunks":
        assert report["policy_chunks"] == 1
        assert len(calls) == 1 + report.get("warmup_queries", 0)
