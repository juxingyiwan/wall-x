"""FIFO session: elapsed time never consumes an action that was only held."""

from dataclasses import dataclass

from .rtc import LatencyBudget, request
from .rtc_queue import ActionQueue, elapsed_steps, guided_request, merge_delay
from .rtc_transitions import Decision, Effects, SessionControl


@dataclass(frozen=True)
class QueueRequest:
    wire: dict
    estimated: int
    index: int


@dataclass(frozen=True)
class QueueContext:
    submitted: float
    observation_age_s: float
    epoch: int
    plan_id: int
    index: int
    guided: bool


class FifoSession(SessionControl[QueueContext]):
    def __init__(self, cfg, started):
        super().__init__(cfg, started)
        self.rtc = cfg.rtc
        self.queue = ActionQueue(cfg.policy.horizon)
        self.latency = LatencyBudget(cfg.rtc)
        self.starved = False
        self.metrics = dict(
            queries=0,
            guided_queries=0,
            bootstrap_queries=0,
            replacements=0,
            discarded_responses=0,
            expired_chunks=0,
            consumption_limited_merges=0,
            queue_starvations=0,
        )

    def camera_lost(self, now):
        self.ensure_live()
        self.generation += 1
        self.queue.clear()
        self.starved = False
        self.next_query = now
        return Effects(Decision.HOLD)

    def toggle_pause(self, now, *, camera_gap=False):
        self.ensure_live()
        self.paused = not self.paused
        if not camera_gap:
            self.camera_lost(now)
        out = Effects(Decision.HOLD)
        out.note("rtc_paused" if self.paused else "rtc_resumed", generation=self.generation)
        return out

    def request_failed(self):
        context = self.take_request()
        out = Effects(Decision.FAULT)
        if context.epoch != self.generation:
            self.metrics["discarded_responses"] += 1
            out.decision = Decision.DISCARD
            out.note("rtc_response_discarded", reason="superseded request error")
        return out

    def receive(self, result, now, camera_generation):
        context = self.take_request()
        out = Effects()
        self.metrics["queries"] += 1
        self.metrics["guided_queries" if context.guided else "bootstrap_queries"] += 1
        stale = (
            self.paused
            or result.generation != self.generation
            or context.epoch != self.generation
            or context.plan_id != self.plan_id
            or result.observation.camera_generation != camera_generation
        )
        duration = now - context.submitted
        elapsed = elapsed_steps(duration, self.rtc.model_dt_s)
        consumed = None if stale else self.queue.index - context.index
        delay = None if stale else merge_delay(elapsed, consumed)
        out.note(
            "rtc_queue_response",
            request_id=result.request_id,
            guided=context.guided,
            latency_s=duration,
            observation_age_s=context.observation_age_s,
            roundtrip_s=result.roundtrip_s,
            elapsed_steps=elapsed,
            consumed_steps=consumed,
            merge_delay_steps=delay,
            stale=stale,
        )
        if stale:
            out.decision = Decision.DISCARD
            self.metrics["discarded_responses"] += 1
            out.note("rtc_response_discarded", reason="superseded RTC generation")
        elif not self.queue.replace(result.actions, delay):
            out.decision = Decision.DISCARD
            if context.guided:
                self.latency.observe(duration)
            self.metrics["expired_chunks"] += 1
            self.metrics["discarded_responses"] += 1
            out.note(
                "rtc_response_discarded",
                reason="no unconsumed raw actions",
                elapsed_steps=elapsed,
                consumed_steps=consumed,
                merge_delay_steps=delay,
                request_id=result.request_id,
            )
        else:
            out.decision = Decision.ACTIVATE
            if context.guided:
                self.latency.observe(duration)
            self.episode_budget.plan_started()
            self.plan_id += 1
            self.active_query = result.request_id
            self.starved = False
            self.metrics["consumption_limited_merges"] += int(delay < elapsed)
            self.metrics["replacements"] += int(context.guided)
            out.note(
                "rtc_queue_replaced",
                request_id=self.active_query,
                plan_id=self.plan_id,
                guided=context.guided,
                first_model_index=delay,
                remaining=len(self.queue),
            )
        wait = min(
            self.rtc.replan_interval_s,
            max(
                0,
                len(self.queue) * self.tick_s - self.latency.estimate_seconds() - self.rtc.guard_s,
            ),
        )
        self.next_query = now + wait
        return out

    def prepare_request(self, now):
        self.ensure_live()
        if self.paused or self.pending is not None or now < self.next_query:
            return None
        if self.episode_budget.chunks >= self.episode_budget.max_policy_chunks:
            if not len(self.queue):
                self.episode_budget.require_plan_capacity()
            return None
        estimated = elapsed_steps(
            self.latency.estimate_seconds() + self.rtc.guard_s, self.rtc.model_dt_s
        )
        wire = (
            guided_request(
                self.rtc,
                self.generation,
                self.plan_id,
                self.queue.leftover(),
                estimated,
                self.cfg.policy.horizon,
            )
            if len(self.queue)
            else request(self.rtc, self.generation, self.plan_id)
        )
        return None if wire is None else QueueRequest(wire, estimated, self.queue.index)

    def submitted(self, proposal, observation, now, submitted):
        # Caller timestamps the actual dispatch after constructing the prefix.
        self.ensure_live()
        if (
            self.paused
            or proposal.wire["epoch"] != self.generation
            or proposal.wire["plan_id"] != self.plan_id
            or proposal.index != self.queue.index
        ):
            raise ValueError("FIFO request proposal is superseded")
        context = QueueContext(
            submitted,
            now - observation.captured_monotonic,
            self.generation,
            self.plan_id,
            proposal.index,
            proposal.wire["prefix"] is not None,
        )
        self.own_request(context)
        out = Effects(Decision.REQUEST)
        out.note(
            "rtc_queue_request",
            submitted_monotonic=submitted,
            observation_monotonic=observation.captured_monotonic,
            available_actions=len(self.queue),
            estimated_delay_steps=proposal.estimated,
            hold_prefix_steps=max(0, proposal.wire["guidance_horizon"] - len(self.queue))
            if context.guided
            else 0,
            **proposal.wire,
        )
        return out

    def pop(self):
        self.ensure_live()
        item = None if self.paused else self.queue.pop()
        out = Effects(Decision.HOLD if item is None else Decision.COMMAND)
        if item is None and not self.paused and self.episode_budget.chunks and not self.starved:
            self.starved = True
            self.metrics["queue_starvations"] += 1
            out.note("rtc_queue_empty", pending=self.pending is not None)
        return item, out

    def stop(self):
        out = super().stop()
        self.queue.clear()
        return out
