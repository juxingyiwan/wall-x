import time
from dataclasses import replace

import numpy as np
import pytest

from wallx_client.config import RTCConfig, validate_config
from wallx_client.contracts import ProtocolError
from wallx_client.rtc import request, validate_ack, validate_capability

DIR = np.array([1, 1, -1, -1, -1, 1, 6.0])
LIMIT = np.array([[-500, 500.0]] * 7)


@pytest.mark.parametrize("seconds", [0.8, 1.5, 3.0])
def test_over_window_budget_is_runtime_fallback_only_when_rtc_enabled(cfg, seconds):
    enabled = replace(cfg, rtc=replace(cfg.rtc, enabled=True, latency_budget_s=seconds))
    validate_config(enabled)
    # Disabled RTC must not validate an unused latency window.
    validate_config(replace(enabled, rtc=replace(enabled.rtc, enabled=False)))


def test_old_server_and_unacknowledged_rtc_are_rejected(cfg):
    with pytest.raises(ProtocolError, match="restart"):
        validate_capability(None, cfg.policy, RTCConfig())
    sent = request(RTCConfig(), 2, 3)
    ack = dict(
        version=1,
        epoch=2,
        plan_id=3,
        committed_steps=0,
        guidance_horizon=0,
        mode="bootstrap",
        guidance_mode="lerobot",
    )
    validate_ack(ack, sent)
    with pytest.raises(ProtocolError):
        validate_ack({**ack, "epoch": 1}, sent)


def test_guidance_mode_negotiation_and_defaults(cfg):
    assert cfg.rtc.guidance_mode == "lerobot" and cfg.rtc.max_guidance == 10
    cap = dict(
        version=1,
        supported=True,
        action_horizon=32,
        real_action_dim=7,
        prefix_units="recorded_commands",
        model_dt_s=1 / 30,
        guidance="endpoint_vjp",
        guidance_modes=["lerobot", "pi_vjp"],
    )
    for mode in cap["guidance_modes"]:
        rtc_cfg = replace(cfg.rtc, guidance_mode=mode)
        validate_capability(cap, cfg.policy, rtc_cfg)
        sent = request(rtc_cfg, 0, 1, np.zeros((32, 7)), 14)
        assert sent["guidance_mode"] == mode
        ack = {
            k: sent[k]
            for k in (
                "version",
                "epoch",
                "plan_id",
                "committed_steps",
                "guidance_horizon",
                "guidance_mode",
            )
        }
        ack["mode"] = "guided"
        validate_ack(ack, sent)
        with pytest.raises(ProtocolError):
            validate_ack({**ack, "guidance_mode": "wrong"}, sent)
        ack.pop("guidance_mode")
        with pytest.raises(ProtocolError):
            validate_ack(ack, sent)
    cap.pop("guidance_modes")
    with pytest.raises(ProtocolError, match="restart"):
        validate_capability(cap, cfg.policy, cfg.rtc)
    with pytest.raises(ValueError, match="guidance_mode"):
        validate_config(replace(cfg, rtc=replace(cfg.rtc, guidance_mode="unknown")))


def test_quit_disarms_with_request_in_flight(cfg, observation, tmp_path):
    import threading
    from types import SimpleNamespace

    from wallx_client.contracts import Prediction
    from wallx_client.rtc_queue_controller import control_loop

    started, disarmed = threading.Event(), threading.Event()
    cfg = replace(cfg, rtc=RTCConfig(enabled=True), control=replace(cfg.control))

    class HW:
        directions = DIR
        limits = LIMIT

        def observe(self):
            now = time.monotonic()
            return replace(
                observation, state=np.zeros(7), captured_monotonic=now, state_monotonic=now
            ), {}

        def hold_command(self, state):
            return state / DIR

        def send(self, raw):
            return {"driver_target_physical": (raw * DIR).tolist()}

        def close(self):
            disarmed.set()

    class Connection:
        def predict(self, obs, task, epoch, rtc_request=None):
            started.set()
            assert disarmed.wait(2)
            return Prediction(
                "x", epoch, np.zeros((32, 7)), obs, time.monotonic(), 0.1, {}, {"mode": "bootstrap"}
            )

        def close(self):
            pass

    recorder = SimpleNamespace(
        path=tmp_path,
        event=lambda *a, **kw: None,
        prediction_async=lambda *a: None,
        check_media=lambda: None,
        sample=lambda *a, **kw: None,
        append=lambda *a, **kw: None,
    )
    report = control_loop(
        HW(), Connection(), cfg, recorder, lambda: "q" if started.is_set() else ""
    )
    assert report["outcome"] == "aborted"
    assert disarmed.is_set()


@pytest.mark.parametrize("fault", ["pause", "feedback"])
def test_rtc_pause_discards_inflight_and_feedback_failure_disarms(
    cfg, observation, tmp_path, fault
):
    import threading
    from types import SimpleNamespace

    from wallx_client.contracts import HardwareError, Prediction
    from wallx_client.rtc_queue_controller import control_loop

    started = threading.Event()
    cfg = replace(
        cfg,
        rtc=RTCConfig(enabled=True),
        control=replace(cfg.control, duration_s=3),
    )
    events = []
    closed = []

    class HW:
        directions = DIR
        limits = LIMIT

        def observe(self):
            if fault == "feedback" and started.is_set():
                raise HardwareError("lost feedback")
            now = time.monotonic()
            return replace(
                observation, state=np.zeros(7), captured_monotonic=now, state_monotonic=now
            ), {}

        def hold_command(self, state):
            return state / DIR

        def send(self, raw):
            return {"driver_target_physical": (raw * DIR).tolist()}

        def close(self):
            closed.append("hardware")

    class Connection:
        def predict(self, obs, task, epoch, rtc_request=None):
            started.set()
            time.sleep(0.15)
            return Prediction(
                "x",
                epoch,
                np.zeros((32, 7)),
                obs,
                time.monotonic(),
                0.15,
                {},
                {"mode": "bootstrap"},
            )

        def close(self):
            closed.append("network")

    recorder = SimpleNamespace(
        path=tmp_path,
        event=lambda kind, **kw: events.append(kind),
        prediction_async=lambda *a: None,
        check_media=lambda: None,
        sample=lambda *a, **kw: None,
        append=lambda *a, **kw: None,
    )

    def key():
        if "rtc_response_discarded" in events:
            return "q"
        if started.is_set() and "rtc_paused" not in events and fault == "pause":
            return "p"
        return ""

    report = control_loop(HW(), Connection(), cfg, recorder, key)
    assert closed[:2] == ["hardware", "network"]
    assert report["outcome"] == ("fault" if fault == "feedback" else "aborted")
    assert "rtc_bootstrap_accepted" not in events
