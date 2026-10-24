import sys
from types import SimpleNamespace

import numpy as np


def test_cabinet_reset_initializes_success_state_without_expert(monkeypatch):
    from wall_x.robotwin.env import RobotwinEnv

    monkeypatch.setitem(
        sys.modules, "envs.utils", SimpleNamespace(ArmTag=lambda name: name)
    )
    task = SimpleNamespace()

    def setup_demo(**kwargs):
        pose = [0.25, 0, 0.82] if kwargs["seed"] == 1 else [-0.25, 0, 0.91]
        task.object = SimpleNamespace(
            get_pose=lambda: SimpleNamespace(p=np.array(pose))
        )

    task.setup_demo = setup_demo
    wrapped = RobotwinEnv({})
    wrapped.env, wrapped.task, wrapped.sim_config = task, "put_object_cabinet", {}
    wrapped._setup(1, 0)
    assert task.arm_tag == "right" and task.origin_z == 0.82
    wrapped._setup(2, 1)
    assert task.arm_tag == "left" and task.origin_z == 0.91
