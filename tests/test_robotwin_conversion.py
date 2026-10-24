"""Validate the stored data boundary without simulator assets or model weights."""

import json

import h5py
import numpy as np
import pytest

from wall_x.robotwin.convert import episode_metadata, read_vectors


def test_hdf5_targets_are_not_shifted_again(tmp_path):
    path = tmp_path / "episode.hdf5"
    with h5py.File(path, "w") as episode:
        episode["data_format_version"] = "v1.0"
        episode["additional_info/frequency"] = 15
        episode["instructions"] = json.dumps(["move bottle", "adjust bottle"])
        for side in ("left", "right"):
            states = np.tile([0, 0, 0, 1, 0, 0, 0], (3, 1)).astype(np.float32)
            states[:, 0] = [1, 2, 3]
            actions = states.copy()
            actions[:, 0] = [2, 3, 4]
            episode[f"state/{side}_ee_poses"] = states
            episode[f"action/{side}_ee_poses"] = actions
            episode[f"state/{side}_ee_joint_states"] = [[0], [0.5], [1]]
            episode[f"action/{side}_ee_joint_states"] = [[0.5], [1], [0]]
        assert episode_metadata(episode) == (15, "move bottle")
        state = read_vectors(episode, "state")
        action = read_vectors(episode, "action")
        np.testing.assert_array_equal(state[:, 0], [1, 2, 3])
        np.testing.assert_array_equal(action[:, 0], [2, 3, 4])
        np.testing.assert_array_equal(action[:, 9], [0.5, 1, 0])


def test_official_codec_handles_both_jpeg_formats():
    codec = pytest.importorskip("XPolicyLab.utils.process_data")
    import cv2

    red = np.zeros((16, 16, 3), dtype=np.uint8)
    red[..., 0] = 255
    marked = codec.encode_image_bit(red)
    ok, legacy = cv2.imencode(".jpg", red)
    assert ok
    for encoded in (marked, legacy.tobytes()):
        decoded = codec.decode_image_bit(encoded)
        assert decoded[..., 0].mean() > 250
        assert decoded[..., 2].mean() < 5
