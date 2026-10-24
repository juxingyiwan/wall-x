"""Independent worker and actor clocks exercise real RTC FIFO control."""

from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest

from wallx_client.config import RTCConfig
from wallx_client.contracts import CameraUnavailable, HardwareError, Observation, Prediction


@pytest.fixture
def simulation(cfg, tmp_path, monkeypatch):
    import wallx_client.camera_recovery as camera_recovery
    import wallx_client.contracts as contracts
    import wallx_client.controller as sync
    import wallx_client.rtc_queue_controller as rq

    events, commands, requests = [], [], []
    options = SimpleNamespace(
        delays=[0.1],
        bootstrap_delay=0.1,
        bootstrap_joint_step=None,
        guided=0,
        feedback_lag=0,
        camera_gap=False,
        feedback_fault=False,
        pause=False,
        paused=False,
        image_age=0.0,
        prefix_error=0.4,
        wake_jitter=0.0,
        observation_delay=0.0,
    )

    class Clock:
        now = 100.0
        worker = None

        def monotonic(self):
            value = self.now if self.worker is None else self.worker
            return value

        time = monotonic

        def sleep(self, seconds):
            self.now += seconds + options.wake_jitter

    clock = Clock()
    for module in (rq, sync, contracts, camera_recovery):
        monkeypatch.setattr(module, "time", clock)

    class Future:
        def __init__(self, value, ready):
            self.value, self.ready = value, ready

        def done(self):
            return clock.now >= self.ready

        def result(self, timeout=None):
            return self.value

        def cancel(self):
            pass

    class Pool:
        def __init__(self, *args, **kw):
            pass

        def submit(self, fn, *args):
            owner = clock.worker is None
            if owner:
                clock.worker = clock.now
            value = fn(*args)
            ready = clock.worker
            if owner:
                clock.worker = None
            return Future(value, ready)

        def shutdown(self, **kw):
            pass

    monkeypatch.setattr(rq, "ThreadPoolExecutor", Pool)
    config = replace(
        cfg,
        rtc=RTCConfig(enabled=True),
        control=replace(cfg.control, duration_s=35),
    )

    class HW:
        directions = np.array([1, 1, -1, -1, -1, 1, 6.0])
        limits = np.array([[-500.0, 500.0]] * 7)
        position = np.zeros(7)
        closed = False
        gap_done = False

        def observe(self):
            clock.now += options.observation_delay
            if options.feedback_fault and len(commands) > 8:
                raise HardwareError("lost feedback")
            if options.camera_gap and options.guided >= 3 and not self.gap_done:
                self.gap_done = True
                raise CameraUnavailable("simulated gap")
            return Observation(
                self.position + options.feedback_lag,
                {n: np.zeros((48, 64, 3), np.uint8) for n in config.policy.camera_names},
                clock.now - options.image_age,
                clock.now,
                clock.now,
                camera_generation=int(self.gap_done),
            ), {}

        def observe_state(self):
            return self.position.copy(), {}

        def hold_command(self, state):
            return self.position / self.directions

        def send(self, raw):
            self.position = raw * self.directions
            return {"driver_target_physical": self.position.tolist()}

        def close(self):
            self.closed = True

    hw = HW()

    class Connection:
        def predict(self, obs, task, epoch, rtc_request=None):
            requests.append(rtc_request)
            guided = rtc_request["prefix"] is not None
            if guided:
                delay = options.delays[options.guided % len(options.delays)]
                options.guided += 1
            else:
                delay = options.bootstrap_delay
            clock.worker += delay
            ack = {
                k: rtc_request[k]
                for k in (
                    "version",
                    "epoch",
                    "plan_id",
                    "committed_steps",
                    "guidance_horizon",
                    "guidance_mode",
                )
            }
            ack["mode"] = "guided" if guided else "bootstrap"
            if guided:
                prefix = np.asarray(rtc_request["prefix"])
                # The policy emits a full chunk even with a shorter RTC input.
                actions = np.tile(prefix[-1], (32, 1))
                actions[: len(prefix)] = prefix
                actions[:, 0] += options.prefix_error
            else:
                actions = np.tile(obs.state / hw.directions, (32, 1))
                actions[:, 0] += np.linspace(0, 2, 32)
                if options.bootstrap_joint_step is not None:
                    actions[0, options.bootstrap_joint_step] += 24
            return Prediction(
                str(clock.monotonic()), epoch, actions, obs, clock.monotonic(), delay, {}, ack
            )

        def close(self):
            pass

    recorder = SimpleNamespace(
        path=tmp_path,
        event=lambda kind, **kw: events.append((kind, dict(at=clock.monotonic(), **kw))),
        prediction_async=lambda *a: None,
        check_media=lambda: None,
        sample=lambda *a, **kw: None,
        append=lambda kind, **kw: commands.append(kw),
    )

    def run(stop, rtc=None):
        selected = replace(config, rtc=rtc or config.rtc)

        def key():
            if stop(events):
                return "q"
            if options.pause and not options.paused and options.guided >= 3:
                options.paused = True
                return "p"
            return ""

        return rq.control_loop(hw, Connection(), selected, recorder, key)

    return SimpleNamespace(
        run=run, options=options, events=events, commands=commands, requests=requests, hw=hw
    )


@pytest.mark.parametrize("latency", [0.1, 0.45, 0.65, 0.82])
def test_raw_queue_multiple_replacements_without_planner(simulation, latency):
    s = simulation
    s.options.delays = [latency]

    r = s.run(
        lambda e: sum(k == "rtc_queue_replaced" and v["guided"] for k, v in e) >= 5,
    )
    assert r["outcome"] == "aborted", (r, s.events)
    assert r["replacements"] == 5
    assert r["target_processing"] == "raw_action_queue"
    assert s.requests[0]["prefix"] is None
    assert any(req["prefix"] is not None for req in s.requests)
    assert all("target_acceleration_deg_s2" not in c for c in s.commands)
    if latency >= 0.65:
        assert r["queue_starvations"] > 0


def test_raw_queue_fully_consumed_chunk_is_discarded_and_bootstrap_recovers(simulation):
    s = simulation
    s.options.delays = [1.5, 0.1]
    r = s.run(
        lambda e: sum(k == "rtc_queue_replaced" and v["guided"] for k, v in e) >= 2,
        rtc=RTCConfig(enabled=True, latency_budget_s=1.5),
    )
    assert r["outcome"] == "aborted", (r, s.events)
    assert r["expired_chunks"] >= 1
    assert r["replacements"] == 2
    assert r["queue_starvations"] >= 1


@pytest.mark.parametrize("latency", [0.65, 0.82, 1.5])
def test_raw_starvation_uses_consumed_index_and_guided_hold_tail(simulation, latency):
    s = simulation
    s.options.delays = [latency]
    r = s.run(
        lambda e: sum(k == "rtc_queue_replaced" and v["guided"] for k, v in e) >= 4,
    )
    assert r["outcome"] == "aborted", r
    responses = [v for k, v in s.events if k == "rtc_queue_response" and not v["stale"]]
    guided = [v for v in responses if v["guided"]]
    assert any(v["elapsed_steps"] > v["consumed_steps"] for v in guided)
    for v in responses:
        assert v["merge_delay_steps"] == min(v["elapsed_steps"], v["consumed_steps"])
        admission = next(
            x
            for k, x in s.events
            if k == "rtc_queue_replaced" and x["request_id"] == v["request_id"]
        )
        assert admission["first_model_index"] == v["merge_delay_steps"]
    padded = [v for k, v in s.events if k == "rtc_queue_request" and v["hold_prefix_steps"]]
    assert padded
    for v in padded:
        p = np.asarray(v["prefix"])
        n = v["available_actions"]
        assert len(p) == v["guidance_horizon"]
        np.testing.assert_array_equal(p[n:], np.tile(p[n - 1], (len(p) - n, 1)))
    assert r["consumption_limited_merges"] > 0


def test_cold_bootstrap_keeps_first_action_even_if_wall_clock_exceeds_chunk(simulation):
    s = simulation
    s.options.bootstrap_delay = 1.5
    r = s.run(lambda e: any(k == "rtc_queue_replaced" for k, _ in e))
    assert r["outcome"] == "aborted", r
    response = next(v for k, v in s.events if k == "rtc_queue_response")
    assert response["elapsed_steps"] >= 45
    assert response["consumed_steps"] == response["merge_delay_steps"] == 0
    first = next(c for c in s.commands if c["phase"] == "rtc_raw")
    assert first["model_action_index"] == 0
    np.testing.assert_array_equal(first["executed_recorded_command"], np.zeros(7))


@pytest.mark.parametrize("joint", [0, 6])
def test_raw_targets_reach_hardware_unchanged(simulation, joint):
    s = simulation
    s.options.bootstrap_joint_step = joint
    r = s.run(lambda e: any(k == "rtc_queue_replaced" for k, _ in e))
    assert r["outcome"] == "aborted", r
    first = next(c for c in s.commands if c["phase"] == "rtc_raw")
    expected = np.zeros(7)
    expected[joint] = 24
    np.testing.assert_array_equal(first["executed_recorded_command"], expected)
    np.testing.assert_array_equal(first["driver_target_physical"], expected * s.hw.directions)
    assert r["policy_commands"] == 1


@pytest.mark.parametrize("fault", ["camera_gap", "pause", "feedback_fault"])
def test_raw_queue_invalidates_pending_on_pause_or_hardware_recovery(simulation, fault):
    s = simulation
    setattr(s.options, fault, True)
    s.options.delays = [0.5]
    r = s.run(lambda e: any(k == "rtc_response_discarded" for k, _ in e))
    assert r["outcome"] == ("fault" if fault == "feedback_fault" else "aborted"), (r, s.events)
    assert s.hw.closed
    if fault != "feedback_fault":
        discarded = [v for k, v in s.events if k == "rtc_queue_response" and v["stale"]]
        assert discarded
        assert not any(
            k == "rtc_queue_replaced" and v["request_id"] == discarded[-1]["request_id"]
            for k, v in s.events
        )


@pytest.mark.parametrize("scenario", ["steady", "jitter", "pause", "camera", "prefix", "feedback"])
def test_frozen_execution_trace(simulation, scenario):
    import hashlib
    import json
    import os
    from pathlib import Path

    s = simulation
    if scenario == "jitter":
        s.options.delays = [0.05, 0.05, 0.64, 0.1, 0.2]
    elif scenario == "pause":
        s.options.pause = True
    elif scenario == "camera":
        s.options.camera_gap = True
    elif scenario == "prefix":
        s.options.prefix_error = 25
    elif scenario == "feedback":
        s.options.feedback_fault = True
    result = s.run(
        lambda e: sum(k == "rtc_queue_replaced" for k, _ in e) >= 5,
    )
    data = {
        "requests": s.requests,
        "commands": s.commands,
        "events": s.events,
        "outcome": result["outcome"],
    }
    encoded = json.dumps(data, sort_keys=True, default=lambda v: v.tolist(), separators=(",", ":"))
    digest = hashlib.sha256(encoded.encode()).hexdigest()
    key = f"{scenario}-fifo"
    directory = Path(
        os.environ.get("WALLX_TRACE_CAPTURE", Path(__file__).parent / "fixtures/rtc_5c8ced1")
    )
    if os.environ.get("WALLX_TRACE_CAPTURE"):
        directory.mkdir(parents=True, exist_ok=True)
        (directory / f"{key}.json").write_text(
            json.dumps(
                {
                    "sha256": digest,
                    "commands": len(s.commands),
                    "requests": len(s.requests),
                    "events": len(s.events),
                    "outcome": result["outcome"],
                }
            )
        )
    else:
        expected = json.loads((directory / f"{key}.json").read_text())
        assert digest == expected["sha256"], (key, digest, expected)
