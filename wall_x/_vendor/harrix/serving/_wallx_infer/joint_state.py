"""Explicit B601 joint protocol for the native Wall-X serving pipeline."""

from pathlib import Path

import numpy as np
from wall_x.inference_padding import validate_padding
from wall_x.checkpoint_weights import checkpoint_sha256, resolve_checkpoint_weights

LAYOUT = {"follow_right_arm_joint_pos": 6, "follow_right_gripper_pos": 1, "action_padding": 19}


def supported(config):
    train = config.train_config
    task = train.get("task", {})
    return all(
        list((train.get(key) or task.get(key, {})).items()) == list(LAYOUT.items())
        for key in ("agent_pos_config", "dof_config")
    )


def ingest_joint_state(state, config, data):
    if not supported(config):
        raise ValueError("joint_state_v1 requires the ordered B601 6+1+19 training layout")
    if set(state) != {"joint_state"}:
        raise ValueError("joint_state cannot be mixed with Cartesian or other state fields")
    value = np.asarray(state["joint_state"], dtype=np.float32)
    if value.shape != (7,) or not np.isfinite(value).all():
        raise ValueError("joint_state must contain seven finite physical motor angles in degrees")
    data.data["state_right_arm_joint_pos"] = value[None, :6].copy()
    data.data["state_right_gripper_pos"] = value[None, 6:].copy()


def joint_metadata(config, *, weights=None):
    if not supported(config):
        return None
    weights = resolve_checkpoint_weights(config.checkpoint_path) if weights is None else Path(weights)
    directory = weights.parent.resolve()
    from wall_x.data.b601 import semantics, validate_checkpoint, digest as manifest_digest
    b601 = semantics(config.train_config)
    extra = {}
    if b601 is not None:
        manifest = validate_checkpoint(config.train_config, directory)
        extra = {"model_action_representation": b601["representation"],
                 "wire_action_representation": b601["wire_action_representation"],
                 "preprocessing_sha256": manifest_digest(manifest),
                 "model_dt_s": 1.0 / manifest["semantics"]["fps"],
                 "model_timebase_source": "validated_b601_manifest"}
    return {
        **extra,
        "protocol_version": 1,
        "padding": validate_padding(getattr(config, "padding", "zero")),
        "state_schema": "joint_state_v1",
        "state_units": "physical_motor_degrees",
        "action_units": "recorded_commands_before_driver_sign_and_gripper_scale",
        "state_dim": 7,
        "action_dim": 26,
        "action_horizon": config.action_horizon,
        "camera_names": list(config.cam_names),
        "checkpoint_id": f"{directory.parent.name}/{directory.name}",
        "checkpoint_sha256": checkpoint_sha256(weights),
        "request_id_echo": True,
    }
