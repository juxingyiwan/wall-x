"""Bind a one-time expert validation to the exact initial policy observation."""

import hashlib
import json
import re

import numpy as np


def observation_digest(observation):
    digest = hashlib.sha256()
    for key in ("state", "face_view", "left_wrist_view", "right_wrist_view"):
        value = np.ascontiguousarray(observation[key])
        digest.update(json.dumps([key, value.dtype.str, value.shape]).encode())
        digest.update(value.tobytes())
    return digest.hexdigest()


def is_prevalidated_scene(row):
    if "expert_validated" not in row and "initial_observation_sha256" not in row:
        return False
    value = row.get("initial_observation_sha256")
    if (
        row.get("expert_validated") is not True
        or not isinstance(value, str)
        or not re.fullmatch("[0-9a-f]{64}", value)
    ):
        raise ValueError("Incomplete expert validation proof in scene manifest")
    return True


def scene_identity(row):
    identity = {key: row[key] for key in ("seed", "instruction")}
    if is_prevalidated_scene(row):
        identity.update(
            expert_validated=True,
            initial_observation_sha256=row["initial_observation_sha256"],
        )
    return identity
