"""Standalone nominal B601 FK for homing only; no CAN or policy dependencies.

The fixed chain describes base_link to gripper_end in the B601 nominal URDF.
Physical motor degrees use the driver's direction and scale mapping.
This geometry does not establish physical TCP calibration.
"""

import numpy as np


def rotation_exp(vector):
    """SO(3) exponential for axis-angle vectors in radians, including zero."""
    vector = np.asarray(vector, dtype=np.float64)
    if vector.shape[-1:] != (3,) or not np.isfinite(vector).all():
        raise ValueError("Expected finite axis-angle vectors of length three")
    x, y, z = np.moveaxis(vector, -1, 0)
    zero = np.zeros_like(x)
    skew = np.stack([zero, -z, y, z, zero, -x, -y, x, zero], -1)
    skew = skew.reshape(*vector.shape[:-1], 3, 3)
    angle = np.linalg.norm(vector, axis=-1)[..., None, None]
    return (
        np.eye(3)
        + np.sinc(angle / np.pi) * skew
        + 0.5 * np.sinc(angle / (2 * np.pi)) ** 2 * (skew @ skew)
    )


_CHAIN = [
    ("joint1", (-0.00034283, -0.00098683, 0.075), (0.0, 0.0, 0.0), (0.0, 0.0, -1.0)),
    ("joint2", (0.020343, 0.027237, 0.07), (-1.5708, 0.0, 0.0), (0.0, 0.0, 1.0)),
    ("joint3", (-0.236, 0.0, 0.0), (0.0, 0.0, 0.0), (0.0, 0.0, -1.0)),
    ("joint4", (0.228, -0.072746, 0.0045), (0.0, 0.0, 0.0), (0.0, 0.0, -1.0)),
    ("joint5", (0.087, -0.048, -0.03075), (-1.5708, 0.0, 0.0), (0.0, 0.0, -1.0)),
    ("joint6", (0.0365, 0.0, 0.048), (0.0, 1.5708, 0.0), (0.0, 0.0, -1.0)),
    ("j_gripper_end", (0.0, 0.0, 0.16621), (3.1416, -1.5708, 0.0), None),
]


class B601HomeGeometry:
    """Nominal base_link -> gripper_end geometry without external model assets."""

    def __init__(self):
        self.config = {
            "base_link": "base_link",
            "tool_link": "gripper_end",
            "joint_names": ["joint1", "joint2", "joint3", "joint4", "joint5", "joint6"],
            "driver_motor_limits_deg": [
                [-145, 145],
                [0, 170],
                [0, 200],
                [-80, 90],
                [-90, 90],
                [-130, 130],
                [0, 270],
            ],
            "urdf_sha256": "2012b5aa3b58878109cb9e3c5deef919a87bd09a67d561662b2904a30dd4397e",
            "calibration_status": "Driver-derived joint mapping and nominal gripper endpoints; "
            "physical zero, axis, TCP and jaw-width calibration are "
            "unverified.",
        }
        self.scale = np.array([1, 1, -1, -1, -1, 1, 6], dtype=float)
        self.offset = np.array([0, 0, 0, 0, 0, 0], dtype=float)
        self.joint_names = tuple(self.config["joint_names"])
        self.chain = []
        for name, xyz, rpy, axis in _CHAIN:
            transform = np.eye(4)
            transform[:3, 3] = xyz
            rotations = rotation_exp(np.eye(3) * np.asarray(rpy)[:, None])
            transform[:3, :3] = rotations[2] @ rotations[1] @ rotations[0]
            if axis is None:
                index = None
            else:
                index = self.joint_names.index(name)
                axis = np.asarray(axis, dtype=float)
                axis /= np.linalg.norm(axis)
            self.chain.append((transform, index, axis))
        self.fk = self

    def __call__(self, joints):
        joints = np.asarray(joints, dtype=np.float64)
        if joints.shape[-1:] != (len(self.joint_names),) or not np.isfinite(joints).all():
            raise ValueError("Expected finite joint vectors in URDF radians")
        pose = np.broadcast_to(np.eye(4), (*joints.shape[:-1], 4, 4)).copy()
        for origin, index, axis in self.chain:
            pose = pose @ origin
            if index is not None:
                pose[..., :3, :3] = pose[..., :3, :3] @ rotation_exp(
                    joints[..., index, None] * axis
                )
        return pose
