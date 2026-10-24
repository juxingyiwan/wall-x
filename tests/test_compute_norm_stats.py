"""Regression: shared DOF names must not overwrite state samples with actions."""

import importlib.util
import os
import sys
from pathlib import Path

import numpy as np

source = Path(os.environ.get("NORM_SCRIPT", "scripts/compute_norm_stats.py"))
spec = importlib.util.spec_from_file_location("compute_stats_under_test", source)
stats = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = stats
spec.loader.exec_module(stats)


def test_shared_joint_names_keep_state_action_stats_independent(monkeypatch, tmp_path):
    states = np.arange(128 * 7, dtype=np.float32).reshape(128, 7) / 10
    actions = states * np.array([1, 1, -1, -1, -1, 1, 1 / 6], dtype=np.float32)
    table = {"observation.state": states, "action": actions}
    monkeypatch.setattr(
        stats,
        "_load_state_action_table",
        lambda *args: (table, "observation.state", "action"),
    )
    ranges = {"follow_right_arm_joint_pos": [0, 6], "follow_right_gripper_pos": [6, 7]}
    result = stats.compute_norm_stats_with_dof_config(
        tmp_path,
        tmp_path / "stats.json",
        ranges,
        ranges,
    )
    for key, array in table.items():
        expected = stats.compute_vector_stats(array)
        for field in ("mean", "std", "q01", "q99"):
            np.testing.assert_allclose(result[key][field], expected[field])


def test_namespaces_allow_same_name_with_different_slices():
    states = np.arange(40, dtype=np.float32).reshape(10, 4)
    actions = np.arange(60, dtype=np.float32).reshape(10, 6) + 100
    result = stats.collect_dof_vectors_from_arrays(
        states,
        actions,
        {"joint": [0, 2]},
        {"joint": [3, 6]},
    )
    np.testing.assert_array_equal(result["state"]["joint"], states[:, :2])
    np.testing.assert_array_equal(result["action"]["joint"], actions[:, 3:6])


def test_relative_action_remains_relative_to_state():
    states = np.arange(24, dtype=np.float32).reshape(8, 3)
    actions = states + 10
    result = stats.collect_dof_vectors_from_arrays(
        states,
        actions,
        {"joint": [0, 3]},
        {"joint_relative": [0, 3]},
        action_chunk=2,
    )
    np.testing.assert_array_equal(result["state"]["joint"], states)
    expected = np.concatenate(
        [actions[offset : offset + 6] - states[:6] for offset in range(2)]
    )
    np.testing.assert_array_equal(result["action"]["joint_relative"], expected)
