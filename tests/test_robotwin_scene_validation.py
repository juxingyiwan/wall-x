"""Direct evaluation rejects unstable expert scenes and surfaces infrastructure errors."""

import sys
from types import ModuleType, SimpleNamespace

import numpy as np
import pytest

from wall_x.robotwin.env import RobotwinEnv
from wall_x.robotwin.panel import observation_digest


@pytest.mark.parametrize(
    "outcome", ["valid", "unstable", "planner_failed", "infrastructure"]
)
def test_direct_expert_scene_validation(monkeypatch, outcome):
    class UnStableError(Exception):
        pass

    for name in (
        "envs",
        "envs.utils",
        "envs.utils.create_actor",
        "generate_episode_instructions",
    ):
        module = ModuleType(name)
        module.UnStableError = UnStableError
        module.generate_episode_descriptions = lambda *args: [
            {"unseen": ["fixed instruction"]}
        ]
        monkeypatch.setitem(sys.modules, name, module)

    closed = []

    def setup(**kwargs):
        if outcome == "unstable":
            raise UnStableError("unstable scene")
        if outcome == "infrastructure":
            raise RuntimeError("renderer failed")

    wrapper = RobotwinEnv({"instruction_count": 100, "instruction_type": "unseen"})
    wrapper.env = SimpleNamespace(
        setup_demo=setup,
        play_once=lambda: {"info": {}},
        plan_success=outcome != "planner_failed",
        check_success=lambda: True,
        close_env=lambda: closed.append(True),
    )
    wrapper.task, wrapper.sim_config = "fixture", {}
    monkeypatch.setattr(wrapper, "_select_task", lambda task: None)
    observation = {
        "state": np.zeros(20, dtype=np.float32),
        **{
            key: np.zeros((2, 2, 3), dtype=np.uint8)
            for key in ("face_view", "left_wrist_view", "right_wrist_view")
        },
    }
    monkeypatch.setattr(wrapper, "_observation", lambda: observation)
    request = dict(task="fixture", seed=100000, index=0)
    if outcome == "infrastructure":
        with pytest.raises(RuntimeError, match="renderer failed"):
            wrapper.check_scene(request)
    elif outcome == "valid":
        result = wrapper.check_scene(request)
        assert result["valid"] and result["expert_validated"]
        assert result["instruction"] == "fixed instruction"
        assert result["initial_observation_sha256"] == observation_digest(observation)
    else:
        assert wrapper.check_scene(request) == {"valid": False}
    assert closed == [True]
