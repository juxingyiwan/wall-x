"""Serial receding-horizon control; the main thread alone commands motors."""

import select
import signal
import sys
import time
from contextlib import contextmanager

import numpy as np

from .camera_recovery import CameraRecovery, record_camera_hold
from .contracts import HardwareError, ProtocolError, TransportError
from .lifecycle import EpisodeBudget, EpisodeLimit, finish_episode
from .recording import write_json
from .workers import DaemonWorker as ThreadPoolExecutor


def check_prediction(prediction, observation, cfg, generation):
    """RTC bootstrap/preflight checks, separate from the shared wire contract."""
    if prediction.observation.camera_generation != observation.camera_generation:
        raise ValueError("Response belongs to an earlier camera stream generation")
    if prediction.generation != generation:
        raise ValueError("Response belongs to an earlier pause/resume generation")
    age = time.monotonic() - prediction.observation.captured_monotonic
    if age > cfg.max_response_age_s:
        raise ValueError(f"Response is {age:.3f}s old; limit is {cfg.max_response_age_s}s")
    drift = np.abs(observation.state - prediction.observation.state)
    if drift[:6].max() > cfg.max_state_drift_arm_deg or drift[6] > cfg.max_state_drift_gripper_deg:
        raise ValueError(f"State changed while inferring: {drift.tolist()}")


@contextmanager
def operator_terminal():
    import termios
    import tty

    if not sys.stdin.isatty():
        raise HardwareError("run must be launched from the on-site operator's interactive terminal")
    fd = sys.stdin.fileno()
    original = termios.tcgetattr(fd)
    previous = {}

    def interrupted(signum, _frame):
        raise KeyboardInterrupt(signal.Signals(signum).name)

    try:
        tty.setcbreak(fd)
        for sig in (signal.SIGINT, signal.SIGTERM, signal.SIGHUP):
            previous[sig] = signal.signal(sig, interrupted)
        yield lambda: sys.stdin.read(1) if select.select([sys.stdin], [], [], 0)[0] else ""
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, original)
        for sig, handler in previous.items():
            signal.signal(sig, handler)


def preflight(hw, connection, cfg, recorder):
    metadata = connection.open()
    write_json(recorder.path / "server.json", metadata)
    recovery = CameraRecovery(cfg.robot.camera_recovery_timeout_s, recorder.event)

    def observe():
        while (value := recovery.poll(lambda: hw.observe(disabled=True))) is None:
            hw.observe_state(disabled=True)
            time.sleep(1 / cfg.control.hz)
        return value[0]

    for attempt in range(1, 4):
        obs = observe()
        obs.validate(cfg.policy.camera_names, cfg.robot.max_camera_age_s)
        recorder.observation(obs, "preflight")
        prediction = connection.predict(obs, cfg.task)
        recorder.prediction(prediction)
        if not cfg.rtc.enabled:
            break
        current = observe()
        try:
            check_prediction(prediction, current, cfg.control, 0)
        except ValueError as exc:
            recorder.event("preflight_response_discarded", attempt=attempt, reason=str(exc))
            if attempt == 3:
                raise
        else:
            break
    recorder.event(
        "preflight_passed",
        roundtrip_s=prediction.roundtrip_s,
        first_recorded_action=prediction.actions[0].tolist(),
    )
    return prediction


def control_loop(hw, connection, cfg, recorder, read_key):
    control = cfg.control
    pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="wallx-inference")
    pending = None
    pending_generation = None
    generation = 0
    chunk = None
    chunk_commands = 0
    action_index = 0
    paused = False
    hold = None
    last_tick = time.monotonic()
    deadline = last_tick
    budget = EpisodeBudget(control, last_tick)
    report = {
        "executor": "sync",
        "outcome": "running",
        "queries": 0,
        "policy_commands": 0,
        "source_actions_completed": 0,
        "discarded_responses": 0,
        "planned_tail_actions": 0,
        "started_wall": time.time(),
        "execution_horizon": control.execution_horizon,
        "target_processing": "native",
    }
    active_query = None
    recovery = CameraRecovery(cfg.robot.camera_recovery_timeout_s, recorder.event)

    def infer(obs, epoch):
        obs.validate(cfg.policy.camera_names, cfg.robot.max_camera_age_s)
        result = connection.predict(obs, cfg.task, epoch)
        recorder.prediction_async(result)
        return result

    print(
        "Running: p=pause/resume, s=success, f=failure, q/Ctrl+C=stop and native shutdown",
        flush=True,
    )
    try:
        while True:
            time.sleep(max(0, deadline - time.monotonic()))
            now = time.monotonic()
            dt = now - last_tick
            last_tick, deadline = now, now + 1 / control.hz
            if dt > control.max_tick_s:
                raise HardwareError(f"Control tick took {dt:.3f}s")
            key = read_key()
            if key in ("s", "f", "q") or (recorder.path / "STOP").exists():
                report["outcome"] = {"s": "success", "f": "failure"}.get(key, "aborted")
                break
            budget.check_time(now)
            value = recovery.poll(hw.observe)
            recorder.check_media()
            if value is None:
                state, status = hw.observe_state()
                if recovery.entered:
                    generation += 1
                    chunk = None
                    hold = hw.hold_command(state)
                if key == "p":
                    paused = not paused
                    recorder.event("paused" if paused else "resumed", generation=generation)
                record_camera_hold(hw, recorder, hold, state, status, generation=generation)
                continue
            observation, status = value
            if hold is None:
                hold = hw.hold_command(observation.state)
            if key == "p":
                generation += 1
                paused = not paused
                chunk = None
                hold = hw.hold_command(observation.state)
                recorder.event("paused" if paused else "resumed", generation=generation)
                print("Paused" if paused else "Resuming from the current observation", flush=True)
            if pending is not None and pending.done():
                future, pending = pending, None
                try:
                    result = future.result()
                except (TransportError, ProtocolError):
                    if pending_generation != generation:
                        report["discarded_responses"] += 1
                        continue
                    chunk = None
                    generation += 1
                    raise  # End the episode; never keep retrying an unavailable server.
                else:
                    report["queries"] += 1
                    if (
                        paused
                        or result.generation != generation
                        or result.observation.camera_generation != observation.camera_generation
                    ):
                        report["discarded_responses"] += 1
                        recorder.event("response_discarded", request_id=result.request_id)
                    else:
                        budget.plan_started()
                        chunk = result.actions[: control.execution_horizon]
                        chunk_commands = 0
                        active_query, action_index = result.request_id, 0
                        report["planned_tail_actions"] += cfg.policy.horizon - len(chunk)
                        recorder.event(
                            "chunk_accepted",
                            request_id=active_query,
                            predicted=cfg.policy.horizon,
                            selected=len(chunk),
                            planned_motion_seconds=len(chunk) / control.hz,
                        )
            phase = "policy" if chunk is not None and not paused else "hold"
            raw = chunk[action_index] if phase == "policy" else hold
            executed = raw
            target = hw.send(executed)
            sent_at = time.monotonic()
            recorder.sample(
                observation,
                action=np.asarray(executed).tolist(),
                physical=target["driver_target_physical"],
                phase=phase,
                query_id=active_query if phase == "policy" else None,
            )
            recorder.append(
                "command",
                phase=phase,
                generation=generation,
                dt_s=dt,
                query_id=active_query if phase == "policy" else None,
                action_index=action_index if phase == "policy" else None,
                raw_recorded_command=np.asarray(raw).tolist(),
                executed_recorded_command=np.asarray(executed).tolist(),
                raw_physical=(np.asarray(raw) * hw.directions).tolist(),
                target_processing="native",
                observed_physical=observation.state.tolist(),
                image_monotonic=observation.captured_monotonic,
                state_monotonic=observation.state_monotonic,
                sent_monotonic=sent_at,
                **status,
                **target,
            )
            if phase == "policy":
                report["policy_commands"] += 1
                chunk_commands += 1
                report["source_actions_completed"] += 1
                action_index += 1
                if action_index == len(chunk):
                    recorder.event(
                        "chunk_completed",
                        request_id=active_query,
                        commands=chunk_commands,
                        source_actions=action_index,
                    )
                    chunk = None
                    hold = np.asarray(target["driver_target_physical"]) / hw.directions
            # Query once per exhausted chunk using this tick's current observation.
            # After a policy tick, acquire again on the next tick; do not require
            # its camera exposure timestamp to postdate the last motor command.
            if not paused and chunk is None and pending is None and phase == "hold":
                budget.require_plan_capacity()
                pending = pool.submit(infer, observation, generation)
                pending_generation = generation
    except EpisodeLimit as exc:
        report.update(
            outcome="timeout" if str(exc) == "duration_s" else "budget_exhausted",
            stop_reason=str(exc),
        )
    except KeyboardInterrupt:
        report["outcome"] = "aborted"
    except Exception as exc:
        report.update(outcome="fault", error=f"{type(exc).__name__}: {exc}")
    finally:
        generation += 1
        chunk = None
        report["policy_chunks"] = budget.chunks
        report["camera_recovery"] = recovery.summary()
        finish_episode(
            hw, connection, report, pending, pool, timeout_s=cfg.shutdown.worker_timeout_s
        )
    return report
