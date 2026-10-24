import threading
import time
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest

from wallx_client.contracts import JOINTS, Prediction, TransportError
from wallx_client.controller import check_prediction, control_loop, preflight
from wallx_client.hardware.b601 import B601
from wallx_client.recording import Recorder


def test_rtc_bootstrap_stale_generation_and_state_drift(cfg, observation):
    pred = Prediction("x", 1, np.zeros((32, 7)), observation, time.monotonic(), 0.01, {})
    with pytest.raises(ValueError, match="generation"):
        check_prediction(pred, observation, cfg.control, 2)
    moved = replace(observation, state=observation.state + 3)
    with pytest.raises(ValueError, match="State changed"):
        check_prediction(pred, moved, cfg.control, 1)
    old = replace(pred, observation=replace(observation, captured_monotonic=time.monotonic() - 3))
    with pytest.raises(ValueError, match="old"):
        check_prediction(old, observation, cfg.control, 1)


def test_driver_direction_and_gripper_scale_applied_exactly_once(cfg):
    hw = B601(cfg.robot, lambda *a, **k: None)
    hw.directions = np.array([1, 1, -1, -1, -1, 1, 6])
    hw.limits = np.array([[-1000, 1000]] * 7)
    received = []

    def send(action):
        received.append(action)
        return {
            f"{j}.pos": action[f"{j}.pos"] * scale
            for j, scale in zip(JOINTS, hw.directions, strict=True)
        }

    hw.robot = SimpleNamespace(send_action=send)
    raw = np.array([1, 2, 3, 4, 5, 6, 7.0])
    result = hw.send(raw)
    assert list(received[0].values()) == raw.tolist()
    assert result["driver_target_physical"] == [1, 2, -3, -4, -5, 6, 42]
    np.testing.assert_array_equal(hw.hold_command(np.array([1, 2, -3, -4, -5, 6, 40.0])), raw)


def test_last_motor_enable_failure_does_not_initiate_home(cfg):
    disabled = []
    hw = B601(cfg.robot, lambda *a, **kw: None)
    hw.enabled = list(JOINTS)  # Last enable was attempted, but did not complete.
    hw.all_enabled = False

    class Motor:
        def __init__(self, name):
            self.name = name

        def disable(self):
            disabled.append(self.name)

        def close(self):
            pass

    class Bus:
        def close_bus(self):
            pass

        def close(self):
            pass

    def forbidden_disconnect():
        raise AssertionError("Partial enable must not home")

    hw.robot = SimpleNamespace(
        bus=Bus(), motors={j: Motor(j) for j in JOINTS}, disconnect=forbidden_disconnect
    )
    hw.close()
    assert disabled == list(reversed(JOINTS))
    assert hw.robot.bus is None


def test_stop_disarms_before_waiting_for_network_worker(cfg, observation):
    started, disarmed = threading.Event(), threading.Event()
    recorder = Recorder(cfg, "test")

    class Hardware:
        directions = np.ones(7)

        def observe(self):
            now = time.monotonic()
            return replace(observation, captured_monotonic=now, state_monotonic=now), {}

        def hold_command(self, state):
            return state

        def send(self, raw):
            return {"driver_target_physical": raw.tolist()}

        def close(self):
            disarmed.set()

    class Connection:
        def predict(self, *args):
            started.set()
            assert disarmed.wait(2), "Main thread waited for network before disarming"
            raise TransportError("closed")

        def close(self):
            pass

    report = control_loop(
        Hardware(), Connection(), cfg, recorder, lambda: "q" if started.is_set() else ""
    )
    recorder.close(report)
    assert report["outcome"] == "aborted"
    assert report["policy_commands"] == 0
    assert disarmed.is_set()


@pytest.mark.parametrize("horizon", [8, 16])
def test_only_selected_action_prefix_steps_are_executed(cfg, observation, horizon):
    cfg = replace(cfg, control=replace(cfg.control, execution_horizon=horizon))
    recorder = Recorder(cfg, "prefix_test")
    executed = []

    class Hardware:
        directions = np.ones(7)

        def observe(self):
            now = time.monotonic()
            return replace(observation, captured_monotonic=now, state_monotonic=now), {}

        def hold_command(self, state):
            return state

        def send(self, raw):
            if raw[0] >= 100:
                executed.append(float(raw[0]))
            return {"driver_target_physical": raw.tolist()}

        def close(self):
            pass

    class Connection:
        def predict(self, obs, instruction, generation):
            actions = np.repeat(np.arange(100, 132, dtype=float)[:, None], 7, axis=1)
            return Prediction("prefix", generation, actions, obs, time.monotonic(), 0.001, {})

        def close(self):
            pass

    result = control_loop(
        Hardware(), Connection(), cfg, recorder, lambda: "s" if len(executed) == horizon else ""
    )
    recorder.close(result)
    assert result["outcome"] == "success"
    assert result["policy_commands"] == horizon
    assert result["planned_tail_actions"] == 32 - horizon
    assert executed == list(range(100, 100 + horizon))


@pytest.mark.parametrize("always_stale", [False, True])
def test_rtc_preflight_uses_fresh_observations_and_bounds_retries(cfg, observation, always_stale):
    cfg = replace(cfg, rtc=replace(cfg.rtc, enabled=True))
    observed = []
    calls = []
    events = []

    class Hardware:
        def observe(self, disabled=False):
            assert disabled
            now = time.monotonic()
            obs = replace(observation, captured_monotonic=now, state_monotonic=now)
            observed.append(obs)
            return obs, {}

    class Connection:
        def open(self):
            return {}

        def predict(self, obs, task):
            calls.append(obs)
            stale = always_stale or len(calls) == 1
            source = replace(obs, captured_monotonic=obs.captured_monotonic - 5) if stale else obs
            return Prediction(
                str(len(calls)), 0, np.zeros((32, 7)), source, time.monotonic(), 0, {}
            )

    recorder = Recorder(cfg, "preflight_retry")
    recorder.observation = lambda *_: None
    recorder.prediction = lambda *_: None
    recorder.event = lambda kind, **kw: events.append(kind)
    if always_stale:
        with pytest.raises(ValueError, match="old"):
            preflight(Hardware(), Connection(), cfg, recorder)
        assert len(calls) == 3
    else:
        preflight(Hardware(), Connection(), cfg, recorder)
        assert len(calls) == 2
        assert events[-1] == "preflight_passed"
    assert len({id(obs) for obs in calls}) == len(calls)
    recorder.close({"outcome": "tested"})


def test_sync_preflight_makes_one_call_without_post_inference_state_gate(cfg, observation):
    calls = []

    class Hardware:
        def observe(self, disabled=False):
            assert disabled
            now = time.monotonic()
            return replace(observation, captured_monotonic=now, state_monotonic=now), {}

    class Connection:
        def open(self):
            return {}

        def predict(self, obs, task):
            calls.append(obs)
            source = replace(obs, captured_monotonic=obs.captured_monotonic - 5)
            return Prediction("preflight", 0, np.zeros((32, 7)), source, time.monotonic(), 0, {})

    recorder = Recorder(cfg, "sync_preflight")
    recorder.observation = lambda *_: None
    try:
        prediction = preflight(Hardware(), Connection(), cfg, recorder)
        assert prediction.request_id == "preflight"
        assert len(calls) == 1
    finally:
        recorder.close({"outcome": "tested"})
