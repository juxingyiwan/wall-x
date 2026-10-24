"""Test the action boundary against Wall-X's actual training transforms."""

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from wall_x.robotwin.codec import (
    observation_to_payload,
    pose_to_state,
    relative_to_pose,
    rotation_6d_to_matrix,
    state_to_pose,
)


def sample_pose(count=8):
    rng = np.random.default_rng(91)
    arms = []
    for _ in range(2):
        quat = Rotation.random(count, random_state=rng).as_quat()[:, [3, 0, 1, 2]]
        arms.append(np.c_[rng.normal(size=(count, 3)), quat, rng.uniform(size=count)])
    return np.concatenate(arms, axis=1)


def assert_same_pose(actual, expected):
    for offset in (0, 8):
        np.testing.assert_allclose(
            actual[..., offset : offset + 3],
            expected[..., offset : offset + 3],
            atol=2e-6,
        )
        np.testing.assert_allclose(
            actual[..., offset + 7], expected[..., offset + 7], atol=2e-6
        )
        # q and -q represent the same rotation.
        dot = np.sum(
            actual[..., offset + 3 : offset + 7]
            * expected[..., offset + 3 : offset + 7],
            axis=-1,
        )
        np.testing.assert_allclose(np.abs(dot), 1, atol=2e-6)


def test_absolute_roundtrip_and_row_layout():
    poses = sample_pose()
    state = pose_to_state(poses)
    assert_same_pose(state_to_pose(state), poses)
    expected = Rotation.from_quat(poses[:, [4, 5, 6, 3]]).as_matrix()
    np.testing.assert_allclose(
        state[:, 3:9], expected[:, :2, :].reshape(-1, 6), atol=1e-7
    )


def test_relative_decode_inverts_training_loader():
    from wall_x.data.backends.lerobot.loader import (
        _compute_delta_from_state_and_abs_rot,
    )

    poses = sample_pose()
    absolute = pose_to_state(poses)
    anchor = absolute[3]
    relative = np.zeros((len(poses), 26))
    for offset in (0, 10):
        relative[:, offset : offset + 3] = (
            absolute[:, offset : offset + 3] - anchor[offset : offset + 3]
        )
        relative[:, offset + 3 : offset + 9] = _compute_delta_from_state_and_abs_rot(
            absolute[:, offset + 3 : offset + 9].astype(np.float64),
            anchor[offset + 3 : offset + 9].astype(np.float64),
        )
        relative[:, offset + 9] = absolute[:, offset + 9]
    relative[:, 20:] = 999  # Virtual padding must never reach the simulator.
    assert_same_pose(relative_to_pose(relative, anchor), poses)


def test_observation_preserves_wxyz_state_and_camera_mapping():
    pose = sample_pose(1)[0]
    rgb = np.zeros((8, 10, 3), dtype=np.uint8)
    rgb[..., 0] = 255
    observation = {
        "endpose": {
            "left_endpose": pose[:7],
            "left_gripper": pose[7],
            "right_endpose": pose[8:15],
            "right_gripper": pose[15],
        },
        "observation": {
            key: {"rgb": rgb.copy()}
            for key in ("head_camera", "left_camera", "right_camera")
        },
    }
    observation["observation"]["left_camera"]["rgb"][..., 1] = 127
    observation["observation"]["right_camera"]["rgb"][..., 2] = 63
    payload = observation_to_payload(observation)
    assert_same_pose(state_to_pose(payload["state"]), pose)
    for source, target in (
        ("head_camera", "face_view"),
        ("left_camera", "left_wrist_view"),
        ("right_camera", "right_wrist_view"),
    ):
        np.testing.assert_array_equal(
            payload[target], observation["observation"][source]["rgb"]
        )


@pytest.mark.parametrize("rotation", [np.zeros(6), np.ones(6), np.full(6, np.nan)])
def test_invalid_rotations_fail_before_simulation(rotation):
    with pytest.raises(ValueError):
        rotation_6d_to_matrix(rotation)
