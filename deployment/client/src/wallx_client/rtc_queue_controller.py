"""RTC execution through a fixed-rate queue of original model actions.

Queue time starts at the producer snapshot/submission, as in the LeRobot actor
example. Camera capture age is logged separately, not used to index this queue.
Only the control thread mutates/consumes the queue and sends motor commands.
"""

import time

from .camera_recovery import CameraRecovery, record_camera_hold
from .contracts import HardwareError, ProtocolError, TransportError
from .lifecycle import EpisodeLimit, finish_episode
from .rtc_fifo_session import FifoSession
from .rtc_transitions import Decision
from .workers import DaemonWorker as ThreadPoolExecutor


def control_loop(hw, connection, cfg, recorder, read_key):
    control = cfg.control
    tick_s = 1 / control.hz
    worker = ThreadPoolExecutor(max_workers=1, thread_name_prefix="wallx-rtc-queue")
    pending = None
    started = last_tick = deadline = time.monotonic()
    session = FifoSession(cfg, started)
    budget = session.episode_budget
    recovery = CameraRecovery(cfg.robot.camera_recovery_timeout_s, recorder.event)
    report = dict(
        executor="rtc_fifo",
        outcome="running",
        rtc=True,
        target_processing="raw_action_queue",
        started_wall=time.time(),
        queries=0,
        guided_queries=0,
        bootstrap_queries=0,
        replacements=0,
        discarded_responses=0,
        expired_chunks=0,
        consumption_limited_merges=0,
        queue_starvations=0,
        policy_commands=0,
        hold_commands=0,
    )

    def infer(obs, epoch, wire):
        obs.validate(cfg.policy.camera_names, cfg.robot.max_camera_age_s)
        result = connection.predict(obs, cfg.task, epoch, rtc_request=wire)
        recorder.prediction_async(result)
        return result

    def emit(effects):
        for notice in effects.notices:
            recorder.event(notice.name, **notice.values)

    def checked_time():
        now = time.monotonic()
        budget.check_time(now)
        if now - last_tick > control.max_tick_s:
            raise HardwareError(f"RTC queue control tick took {now - last_tick:.3f}s")
        return now

    print(
        "RTC FIFO executes model actions at the control rate. "
        "Empty queue holds the last target. p=pause, q=normal stop (homes).",
        flush=True,
    )
    try:
        while True:
            time.sleep(max(0, deadline - time.monotonic()))
            now = time.monotonic()
            dt, last_tick, deadline = now - last_tick, now, now + tick_s
            if dt > control.max_tick_s:
                raise HardwareError(f"RTC queue control tick took {dt:.3f}s")
            key = read_key()
            if key in ("s", "f", "q") or (recorder.path / "STOP").exists():
                report["outcome"] = {"s": "success", "f": "failure"}.get(key, "aborted")
                break
            budget.check_time(now)
            value = recovery.poll(hw.observe)
            recorder.check_media()
            now = checked_time()
            if value is None:
                state, status = hw.observe_state()
                if recovery.entered:
                    session.camera_lost(now)
                if key == "p":
                    emit(session.toggle_pause(now, camera_gap=True))
                record_camera_hold(
                    hw,
                    recorder,
                    hw.hold_command(state),
                    state,
                    status,
                    generation=session.generation,
                )
                continue
            obs, status = value
            if key == "p":
                emit(session.toggle_pause(now))
            if pending is not None and pending.done():
                future, pending = pending, None
                try:
                    result = future.result()
                except (TransportError, ProtocolError):
                    effects = session.request_failed()
                    emit(effects)
                    if effects.decision == Decision.FAULT:
                        raise
                else:
                    now = checked_time()
                    emit(session.receive(result, now, obs.camera_generation))
            now = checked_time()
            proposal = session.prepare_request(now)
            if proposal is not None:
                effects = session.submitted(proposal, obs, now, time.monotonic())
                pending = worker.submit(infer, obs, session.generation, proposal.wire)
                emit(effects)
            checked_time()
            item, effects = session.pop()
            emit(effects)
            if item is None:
                raw, model_index = hw.hold_command(obs.state), None
                report["hold_commands"] += 1
                phase = "hold"
            else:
                raw, model_index = item
                phase = "rtc_raw"
            target = hw.send(raw)
            report["policy_commands"] += int(item is not None)
            recorder.sample(
                obs,
                action=raw.tolist(),
                physical=target["driver_target_physical"],
                phase=phase,
                query_id=session.active_query if item is not None else None,
            )
            recorder.append(
                "command",
                phase=phase,
                generation=session.generation,
                plan_id=session.plan_id,
                query_id=session.active_query if item is not None else None,
                model_action_index=model_index,
                queue_remaining=len(session.queue),
                dt_s=dt,
                target_processing="raw_action_queue",
                executed_recorded_command=raw.tolist(),
                observed_physical=obs.state.tolist(),
                image_monotonic=obs.captured_monotonic,
                state_monotonic=obs.state_monotonic,
                sent_monotonic=time.monotonic(),
                **status,
                **target,
            )
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
        session.stop()
        report.update(session.metrics)
        report["policy_chunks"] = budget.chunks
        report["camera_recovery"] = recovery.summary()
        finish_episode(
            hw, connection, report, pending, worker, timeout_s=cfg.shutdown.worker_timeout_s
        )
    return report
