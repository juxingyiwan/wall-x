"""Regression: shared DOF names must not overwrite state samples with actions."""

import numpy as np

from wall_x.robotwin import norm_stats as stats


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


def test_vector_stats_float32_constant_accumulation():
    # A multi-column float32 reduction drifts even for constant rotations.
    constants = np.array([0.97, -0.97, 0.95, -0.95], dtype=np.float32)
    values = np.tile(constants, (100_000, 1))
    result = stats.compute_vector_stats(values)
    np.testing.assert_allclose(
        result["mean"], constants.astype(np.float64), rtol=0, atol=1e-12
    )
    np.testing.assert_allclose(result["std"], np.zeros(4), rtol=0, atol=1e-12)
    np.testing.assert_array_equal(result["q01"], constants)
    np.testing.assert_array_equal(result["q99"], constants)
    assert values.dtype == np.float32
    np.testing.assert_array_equal(values[0], constants)
    np.testing.assert_array_equal(values[-1], constants)


def test_vector_stats_small_variance_with_large_offset():
    # Exact binary fractions give an analytic mean and standard deviation.
    values = np.tile(
        np.array([[10000.0, 10000.125], [10000.125, 10000.0]], dtype=np.float32),
        (32768, 1),
    )
    result = stats.compute_vector_stats(values)
    np.testing.assert_allclose(
        result["mean"], [10000.0625, 10000.0625], rtol=0, atol=1e-12
    )
    np.testing.assert_allclose(result["std"], [0.0625, 0.0625], rtol=0, atol=1e-12)
    np.testing.assert_array_equal(result["q01"], [10000.0, 10000.0])
    np.testing.assert_array_equal(result["q99"], [10000.125, 10000.125])
