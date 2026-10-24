"""Tracing preserves control results and records transient contacts/planner failures."""

import json
from types import SimpleNamespace

import numpy as np
import pytest

from wall_x.robotwin.diagnostics import RolloutTrace


@pytest.mark.parametrize("fail", [False, True])
def test_trace_preserves_methods_and_intermediate_contact(tmp_path, fail):
    class Robot:
        def left_plan_path(self, action):
            return {"status": "Failure"}

        def right_plan_path(self, action):
            return {"status": "Success", "position": [action]}

        def get_left_gripper_val(self):
            return 0.0

        def get_right_gripper_val(self):
            return 0.0

    class Environment:
        def __init__(self):
            self.robot = Robot()
            self.take_action_cnt = 0
            self.eval_success = False
            self.bell_id = 1
            self.bell = SimpleNamespace(
                get_contact_point=lambda _: np.zeros(3),
                get_pose=lambda: SimpleNamespace(p=[-0.1, 0, 0]),
            )
            self.contacts = []
            self.checks = 0

        def get_arm_pose(self, arm):
            return [0, 0, 0, 1, 0, 0, 0]

        def check_arm_function(self):
            return True

        def check_success(self):
            self.checks += 1
            return bool(self.contacts)

        def get_gripper_actor_contact_position(self, name):
            return self.contacts

        def take_action(self, action, action_type):
            self.robot.left_plan_path(action)
            self.robot.right_plan_path(action)
            self.contacts = [np.array([0.01, 0, 0])]
            assert self.check_success()
            self.contacts = []
            assert not self.check_success()
            self.take_action_cnt += 1
            if fail:
                raise RuntimeError("simulation failed")

    env = Environment()
    path = tmp_path / "trace.jsonl"

    def run():
        with RolloutTrace(env, path) as trace:
            trace.take_action(np.zeros(16))

    if fail:
        with pytest.raises(RuntimeError, match="simulation failed"):
            run()
    else:
        run()
    assert env.checks == 2
    assert "check_success" not in vars(env)
    assert "left_plan_path" not in vars(env.robot)
    assert "right_plan_path" not in vars(env.robot)
    row = json.loads(path.read_text().splitlines()[1])
    assert row["planner"]["left"]["status"] == "Failure"
    assert row["planner"]["right"]["path_steps"] == 1
    assert row["physics_checks"] == row["closed_checks"] == 2
    assert row["contact_points"] == row["closed_target_contacts"] == 1
    assert row["nearest_contact"]["distance"] == pytest.approx(0.01)
