"""Drive real session transitions using numbers, without clocks or worker mocks."""

from dataclasses import replace

import numpy as np
import pytest

from wallx_client.config import RTCConfig
from wallx_client.contracts import Observation, Prediction
from wallx_client.rtc_fifo_session import FifoSession
from wallx_client.rtc_transitions import Decision


@pytest.fixture
def config(cfg):
    return replace(
        cfg,
        rtc=RTCConfig(enabled=True),
        control=replace(cfg.control, duration_s=30),
    )


def observation(at, generation=0):
    return Observation(np.zeros(7), {}, at, 0, at, camera_generation=generation)


def prediction(obs, at, epoch=0, *, actions=None, guided=False):
    if actions is None:
        actions = np.repeat(np.arange(32)[:, None], 7, axis=1).astype(float)
    return Prediction(
        f"request-{obs.state_monotonic}",
        epoch,
        actions,
        obs,
        at,
        0.1,
        {},
        rtc={"mode": "guided" if guided else "bootstrap"},
    )


def started_fifo(config):
    session = FifoSession(config, 10)
    obs = observation(10)
    proposal = session.prepare_request(10)
    session.submitted(proposal, obs, 10, 10)
    effects = session.receive(prediction(obs, 10.5), 10.5, 0)
    assert effects.decision == Decision.ACTIVATE
    # Bootstrap never skips 15 actions merely because the model took 0.5 s.
    assert effects.notices[0].values["merge_delay_steps"] == 0
    return session


def test_fifo_merge_is_bounded_by_actual_consumption(config):
    session = started_fifo(config)
    now = session.next_query
    obs = observation(now)
    proposal = session.prepare_request(now)
    session.submitted(proposal, obs, now, now)
    for _ in range(12):
        assert session.pop()[0] is not None
    result = prediction(
        obs, now + 0.54, actions=np.full((32, 7), 100) + np.arange(32)[:, None], guided=True
    )
    effects = session.receive(result, now + 0.54, 0)
    response = effects.notices[0].values
    assert response["elapsed_steps"] > 12
    assert response["consumed_steps"] == response["merge_delay_steps"] == 12
    np.testing.assert_array_equal(session.pop()[0][0], np.full(7, 112))


def test_fifo_hold_never_consumes_and_expired_response_never_replaces(config):
    session = started_fifo(config)
    now = session.next_query
    obs = observation(now)
    session.submitted(session.prepare_request(now), obs, now, now)
    for _ in range(32):
        session.pop()
    index = session.queue.index
    for _ in range(6):
        assert session.pop()[0] is None
    assert session.queue.index == index
    effects = session.receive(prediction(obs, now + 2, guided=True), now + 2, 0)
    assert effects.decision == Decision.DISCARD and not len(session.queue)
    assert session.metrics["expired_chunks"] == session.metrics["queue_starvations"] == 1


@pytest.mark.parametrize("superseded", [False, True])
def test_request_error_is_fault_only_for_current_generation(config, superseded):
    session = started_fifo(config)
    at = session.next_query
    session.submitted(session.prepare_request(at), observation(at), at, at)
    if superseded:
        session.toggle_pause(at)
    assert session.request_failed().decision == (Decision.DISCARD if superseded else Decision.FAULT)
    assert session.pending is None


def test_stop_invalidates_requests_and_is_terminal(config):
    session = started_fifo(config)
    at = session.next_query
    session.submitted(session.prepare_request(at), observation(at), at, at)
    assert session.stop().decision == Decision.STOP
    generation = session.generation
    session.stop()
    assert session.generation == generation and session.pending is None
    with pytest.raises(RuntimeError, match="stopped"):
        session.request_failed()
    with pytest.raises(RuntimeError, match="stopped"):
        session.camera_lost(at + 1)


def test_fifo_pause_invalidates_pending_without_consuming_hold(config):
    session = started_fifo(config)
    at = session.next_query
    obs = observation(at)
    session.submitted(session.prepare_request(at), obs, at, at)
    session.toggle_pause(at + 0.01)
    assert session.pop()[0] is None
    effects = session.receive(prediction(obs, at + 0.2, guided=True), at + 0.2, 0)
    assert effects.decision == Decision.DISCARD and not len(session.queue)
    session.toggle_pause(at + 0.3)
    assert session.prepare_request(at + 0.3).wire["prefix"] is None


def test_fifo_rejects_dispatch_proposal_invalidated_by_pause(config):
    session = started_fifo(config)
    at = session.next_query
    proposal = session.prepare_request(at)
    session.toggle_pause(at)
    with pytest.raises(ValueError, match="superseded"):
        session.submitted(proposal, observation(at), at, at)
    assert session.pending is None
