"""Raw-queue starvation regressions and LeRobot-compatible hold prefixes."""

import numpy as np
import pytest

from wallx_client.config import RTCConfig
from wallx_client.rtc_queue import (
    ActionQueue,
    elapsed_steps,
    guided_request,
    merge_delay,
)


def test_queue_sends_exact_original_suffix_and_retains_original_leftover():
    actions = np.arange(32 * 7, dtype=np.float32).reshape(32, 7)
    q = ActionQueue(32)
    assert q.replace(actions, 23)
    actions[:] = -999
    assert len(q) == 9
    remaining = q.leftover()
    remaining[:] = -123
    for index in range(23, 32):
        value, source_index = q.pop()
        assert source_index == index
        np.testing.assert_array_equal(value, np.arange(index * 7, (index + 1) * 7))
    assert q.pop() is None and len(q) == 0


def test_expired_chunk_never_rewinds_or_erases_existing_queue():
    q = ActionQueue(32)
    original = np.ones((32, 7), np.float32)
    q.replace(original, 0)
    q.pop()
    assert not q.replace(original * 2, 32)
    np.testing.assert_array_equal(q.leftover(), original[1:])
    assert not q.replace(original * 3, 75)
    assert q.pop()[1] == 1


@pytest.mark.parametrize("available", [1, 2, 6, 9, 20, 31, 32])
@pytest.mark.parametrize("estimate", [1, 14, 26, 50])
def test_short_prefix_guides_the_hold_after_original_actions(available, estimate):
    old = np.arange(available * 7, dtype=np.float32).reshape(available, 7) + 1
    req = guided_request(RTCConfig(), 3, 5, old, estimate, 32)
    prefix = np.asarray(req["prefix"])
    end = req["guidance_horizon"]
    assert prefix.shape == (end, 7)
    count = min(available, end)
    np.testing.assert_array_equal(prefix[:count], old[:count])
    np.testing.assert_array_equal(prefix[count:], np.tile(old[-1], (end - count, 1)))
    assert 1 <= req["committed_steps"] < req["guidance_horizon"] < 32
    assert req["committed_steps"] == min(estimate, 30)
    assert req["guidance_horizon"] == min(min(estimate, 30) + 6, 31)
    assert req["epoch"] == 3 and req["plan_id"] == 5
    prefix[:] = -999
    assert (old > 0).all()


def test_empty_prefix_cannot_invent_a_hold_pose():
    assert guided_request(RTCConfig(), 0, 1, np.zeros((0, 7)), 26, 32) is None


def test_elapsed_latency_includes_holds_but_merge_does_not_skip_unexecuted_motion():
    q = ActionQueue(32)
    q.replace(np.zeros((32, 7)), 23)
    for _ in range(9):
        q.pop()
    assert q.index == 9
    # Latency remains 23 steps for future guidance. Only nine actions actually
    # left the queue; holding for the rest must not splice ahead of the robot.
    assert elapsed_steps(0.76, 1 / 30) == 23
    assert merge_delay(23, q.index) == 9
    assert elapsed_steps(1.5, 1 / 30) == 45
    assert elapsed_steps(14 / 30, 1 / 30) == 14


@pytest.mark.parametrize(
    "elapsed,consumed,expected",
    [(21, 13, 13), (20, 20, 20), (20, 21, 20), (45, 0, 0), (45, 32, 32)],
)
def test_merge_matches_upstream_consumption_bound(elapsed, consumed, expected):
    assert merge_delay(elapsed, consumed) == expected


@pytest.mark.parametrize("elapsed,consumed", [(-1, 3), (1, -1), (1.5, 2), (True, 0)])
def test_invalid_consumption_counts_fail_closed(elapsed, consumed):
    with pytest.raises(ValueError, match="merge counts"):
        merge_delay(elapsed, consumed)


def test_recorded_shoulder_jump_does_not_skip_unexecuted_free_suffix():
    # Numeric regression from 2026-10-09: 13 consumed, 21 elapsed, old final
    # target 37.6598 deg. No images, robot access or rollout files are required.
    old = np.zeros((32, 7), np.float32)
    old[:, 1] = 37.659793853759766
    new = old.copy()
    new[13, 1] = 42.30813980102539
    new[21, 1] = 61.895599365234375
    q = ActionQueue(32)
    q.replace(old, 19)
    before = q.index
    for _ in range(21):
        q.pop()
    delay = merge_delay(21, q.index - before)
    q.replace(new, delay)
    action, index = q.pop()
    assert index == 13
    assert float(action[1] - old[-1, 1]) == pytest.approx(4.64834595)
    # Avoiding this recorded 24-degree skip is not a general motion bound.
    assert float(new[21, 1] - old[-1, 1]) > 24
