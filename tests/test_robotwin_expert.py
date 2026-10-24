"""Keep expected expert rejections separate from programming failures."""

from types import SimpleNamespace

import pytest

from wall_x.robotwin.expert import (
    install_failed_grasp_guard,
    install_unreachable_pose_guard,
)


class InvalidSeed(Exception):
    pass


def test_failed_dustbin_grasp_is_rejected():
    env = SimpleNamespace(plan_success=False, grasp_actor=lambda: (None, []))
    install_failed_grasp_guard(env, "put_bottles_dustbin", InvalidSeed)
    with pytest.raises(InvalidSeed, match="planning failed"):
        env.grasp_actor()


@pytest.mark.parametrize(
    "task,success",
    [
        ("put_bottles_dustbin", True),
        ("open_microwave", False),
    ],
)
def test_other_cases_keep_original_result(task, success):
    result = (None, [])
    env = SimpleNamespace(plan_success=success, grasp_actor=lambda: result)
    install_failed_grasp_guard(env, task, InvalidSeed)
    assert env.grasp_actor() is result


def test_valid_actions_and_unrelated_errors_are_preserved():
    actions = ("left", [object(), object()])
    env = SimpleNamespace(plan_success=True, grasp_actor=lambda: actions)
    install_failed_grasp_guard(env, "put_bottles_dustbin", InvalidSeed)
    assert env.grasp_actor() is actions

    def fail():
        raise IndexError("unrelated upstream bug")

    broken = SimpleNamespace(plan_success=False, grasp_actor=fail)
    install_failed_grasp_guard(broken, "put_bottles_dustbin", InvalidSeed)
    with pytest.raises(IndexError, match="unrelated"):
        broken.grasp_actor()


def test_unreachable_alarmclock_pose_is_an_expert_rejection():
    env = SimpleNamespace(plan_success=True, choose_best_pose=lambda *a, **kw: None)
    install_unreachable_pose_guard(env, "click_alarmclock", InvalidSeed)
    with pytest.raises(InvalidSeed, match="no reachable alarm-clock pose"):
        env.choose_best_pose([0] * 7, [0] * 7, arm_tag="left")


@pytest.mark.parametrize("pose", [[0, 0, 1, 1, 0, 0, 0], [-1] * 7])
def test_alarmclock_valid_pose_and_existing_failure_sentinel_unchanged(pose):
    calls = []

    def original(*args, **kwargs):
        calls.append((args, kwargs))
        return pose

    env = SimpleNamespace(choose_best_pose=original)
    install_unreachable_pose_guard(env, "click_alarmclock", InvalidSeed)
    wrapped = env.choose_best_pose
    install_unreachable_pose_guard(env, "click_alarmclock", InvalidSeed)
    assert env.choose_best_pose is wrapped
    assert env.choose_best_pose("target", arm_tag="right") is pose
    assert calls == [(("target",), {"arm_tag": "right"})]


def test_unreachable_pose_guard_does_not_mask_other_tasks_or_errors():
    other = SimpleNamespace(choose_best_pose=lambda: None)
    install_unreachable_pose_guard(other, "adjust_bottle", InvalidSeed)
    assert other.choose_best_pose() is None

    def broken():
        raise TypeError("unrelated planner programming error")

    env = SimpleNamespace(choose_best_pose=broken)
    install_unreachable_pose_guard(env, "click_alarmclock", InvalidSeed)
    with pytest.raises(TypeError, match="unrelated"):
        env.choose_best_pose()
