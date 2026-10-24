"""RoboTwin absolute WXYZ poses <-> Wall-X row-based 6D rotations.

Dataset vectors contain 20 absolute values (10 per arm). The training loader
forms chunk-anchor-relative poses and appends six virtual padding values.
The model output is already unnormalized; only the relative transform is
undone here. Grippers remain absolute, with 0 closed and 1 open.
"""

import numpy as np
from scipy.spatial.transform import Rotation

CAMERAS = {
    "cam_head": ("head_camera", "face_view"),
    "cam_left_wrist": ("left_camera", "left_wrist_view"),
    "cam_right_wrist": ("right_camera", "right_wrist_view"),
}


def _vector(value, width, name):
    array = np.asarray(value, dtype=np.float64)
    if array.ndim < 1 or array.shape[-1] != width or not np.isfinite(array).all():
        raise ValueError(f"{name} must have finite values and last dimension {width}")
    return array


def rotation_6d_to_matrix(value):
    value = _vector(value, 6, "rotation")
    first, second = value[..., :3], value[..., 3:]
    norm = np.linalg.norm(first, axis=-1, keepdims=True)
    if np.any(norm < 1e-8):
        raise ValueError("Degenerate first rotation row")
    first = first / norm
    second = second - (first * second).sum(-1, keepdims=True) * first
    norm = np.linalg.norm(second, axis=-1, keepdims=True)
    if np.any(norm < 1e-8):
        raise ValueError("Collinear rotation rows")
    second = second / norm
    return np.stack((first, second, np.cross(first, second)), axis=-2)


def pose_to_state(pose):
    """Convert [...,16] [xyz,WXYZ,grip] x 2 into absolute [...,20]."""
    pose = _vector(pose, 16, "RoboTwin pose")
    arms = []
    for offset in (0, 8):
        arm = pose[..., offset : offset + 8]
        quat = arm[..., [4, 5, 6, 3]]  # SciPy XYZW
        if np.any(np.linalg.norm(quat, axis=-1) < 1e-8):
            raise ValueError("Zero quaternion")
        matrix = Rotation.from_quat(quat.reshape(-1, 4)).as_matrix()
        six = matrix[:, :2, :].reshape(arm.shape[:-1] + (6,))
        arms.append(np.concatenate((arm[..., :3], six, arm[..., 7:8]), axis=-1))
    return np.concatenate(arms, axis=-1).astype(np.float32)


def state_to_pose(state):
    """Convert absolute 20D values back to simulator pose commands."""
    state = _vector(state, 20, "Wall-X state")
    arms = []
    for offset in (0, 10):
        arm = state[..., offset : offset + 10]
        matrix = rotation_6d_to_matrix(arm[..., 3:9])
        quat = Rotation.from_matrix(matrix.reshape(-1, 3, 3)).as_quat()
        quat = quat[:, [3, 0, 1, 2]].reshape(arm.shape[:-1] + (4,))
        arms.append(np.concatenate((arm[..., :3], quat, arm[..., 9:10]), axis=-1))
    return np.concatenate(arms, axis=-1)


def relative_to_pose(actions, anchor):
    """Decode an unnormalized [H,26] chunk using the state at inference time."""
    actions = _vector(actions, 26, "Model action chunk")
    anchor = _vector(anchor, 20, "Chunk anchor")
    if actions.ndim != 2 or anchor.ndim != 1:
        raise ValueError("Expected actions [H,26] and anchor [20]")
    absolute = actions[:, :20].copy()
    for offset in (0, 10):
        absolute[:, offset : offset + 3] += anchor[offset : offset + 3]
        rotation = rotation_6d_to_matrix(actions[:, offset + 3 : offset + 9])
        rotation = rotation @ rotation_6d_to_matrix(anchor[offset + 3 : offset + 9])
        absolute[:, offset + 3 : offset + 9] = rotation[:, :2, :].reshape(-1, 6)
        absolute[:, offset + 9] = np.clip(absolute[:, offset + 9], 0, 1)
    return state_to_pose(absolute)


def observation_to_payload(observation):
    """Compact simulator observation for direct, in-process model inference."""
    endpose = observation["endpose"]
    pose = np.concatenate(
        (
            endpose["left_endpose"],
            [endpose["left_gripper"]],
            endpose["right_endpose"],
            [endpose["right_gripper"]],
        )
    )
    state = pose_to_state(pose)
    views = {}
    for simulator_key, model_key in CAMERAS.values():
        rgb = np.asarray(observation["observation"][simulator_key]["rgb"])
        if rgb.dtype != np.uint8 or rgb.ndim != 3 or rgb.shape[-1] != 3:
            raise ValueError(f"Expected uint8 RGB image from {simulator_key}")
        views[model_key] = rgb
    return {"state": state, **views}
