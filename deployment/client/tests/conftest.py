import time
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from wallx_client.config import load_config
from wallx_client.contracts import ACTION_UNITS, STATE_UNITS, Observation


@pytest.fixture
def cfg(tmp_path):
    source = Path(__file__).resolve().parents[1] / "src/wallx_client/configs/b601.example.yml"
    value = load_config(source)
    return replace(
        value,
        policy=replace(value.policy, checkpoint_sha256="a" * 64),
        output_root=str(tmp_path / "rollouts"),
        robot=replace(value.robot, camera_mapping_verified=False),
        # Keep synchronous-controller fixtures independent of deployment defaults.
        control=replace(value.control, execution_horizon=8),
    )


@pytest.fixture
def observation():
    now = time.monotonic()
    return Observation(
        np.arange(7, dtype=np.float32),
        {
            k: np.full((48, 64, 3), [255, 30, 15], dtype=np.uint8)
            for k in ("front_view", "side_view")
        },
        now,
        time.time(),
        now,
    )


@pytest.fixture
def metadata(cfg):
    return {
        "wallx_client": {
            "protocol_version": 1,
            "state_schema": "joint_state_v1",
            "state_units": STATE_UNITS,
            "action_units": ACTION_UNITS,
            "state_dim": 7,
            "action_dim": 26,
            "action_horizon": 32,
            "camera_names": ["front_view", "side_view"],
            "checkpoint_id": cfg.policy.checkpoint_id,
            "checkpoint_sha256": cfg.policy.checkpoint_sha256,
            "request_id_echo": True,
            "padding": cfg.policy.padding,
        }
    }
