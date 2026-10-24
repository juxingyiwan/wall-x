"""Narrow runtime guards for upstream RoboTwin expert seed failures."""

from functools import wraps


def install_failed_grasp_guard(env, task, invalid_seed_error):
    """Reject the upstream failed-plan sentinel before dustbin code indexes it.

    This task accesses the grasp action list after planning has already failed.
    Base_Task.grasp_actor then returns (None, []), which cannot form a valid
    demonstration. Other tasks may intentionally recover from failed plans;
    their behavior and all successful grasp results are left untouched.
    """
    if task != "put_bottles_dustbin" or getattr(
        env, "_wallx_failed_grasp_guard", False
    ):
        return
    original = env.grasp_actor

    @wraps(original)
    def checked(*args, **kwargs):
        result = original(*args, **kwargs)
        if (
            env.plan_success is False
            and isinstance(result, tuple)
            and len(result) == 2
            and result[0] is None
            and isinstance(result[1], list)
            and not result[1]
        ):
            raise invalid_seed_error("Expert planning failed before a dustbin grasp")
        return result

    env.grasp_actor = checked
    env._wallx_failed_grasp_guard = True


def install_unreachable_pose_guard(env, task, invalid_seed_error):
    """Reject alarm-clock expert scenes with no reachable candidate pose.

    Upstream choose_best_pose returns None when no candidate plan succeeds;
    click_alarmclock immediately slices that result. Guard this precise return
    value, rather than swallowing arbitrary TypeErrors or missing asset data.
    Other tasks may handle None and recover, so keep their behavior unchanged.
    """
    if task != "click_alarmclock" or getattr(
        env, "_wallx_unreachable_pose_guard", False
    ):
        return
    original = env.choose_best_pose

    @wraps(original)
    def checked(*args, **kwargs):
        pose = original(*args, **kwargs)
        if pose is None:
            raise invalid_seed_error(
                "Expert planning found no reachable alarm-clock pose"
            )
        return pose

    env.choose_best_pose = checked
    env._wallx_unreachable_pose_guard = True
