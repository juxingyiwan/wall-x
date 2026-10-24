import numpy as np
import pytest

from wall_x.robotwin.convert_lerobot import convert_vectors


def test_wxyz_rotation_and_already_aligned_action_contract():
    # A +90 degree Z rotation must produce row-6D [0,-1,0,1,0,0].
    arm = [1.0, 2.0, 3.0, 2**-0.5, 0.0, 0.0, 2**-0.5, 0.75]
    state = np.array([arm + arm, arm + arm], dtype=np.float32)
    action = state.copy()
    action[:, 0] = [10.0, 20.0]
    action[:, 7] = [0.1, 0.2]
    converted_state, converted_action = convert_vectors(state, action)
    np.testing.assert_allclose(
        converted_state[:, 3:9], [[0, -1, 0, 1, 0, 0]] * 2, atol=1e-6
    )
    np.testing.assert_array_equal(converted_action[:, 0], [10.0, 20.0])
    np.testing.assert_allclose(converted_action[:, 9], [0.1, 0.2])
    np.testing.assert_array_equal(converted_state[:, :3], [[1, 2, 3]] * 2)


@pytest.mark.parametrize("index,value", [(3, 0), (7, 2), (0, np.nan)])
def test_invalid_source_vectors_fail_before_training(index, value):
    arm = [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]
    state = np.array([arm + arm], dtype=np.float32)
    state[0, index] = value
    with pytest.raises(ValueError):
        convert_vectors(state, state.copy())
