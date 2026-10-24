import base64
from dataclasses import replace

import cv2
import msgpack
import msgpack_numpy
import numpy as np
import pytest

from wallx_client.contracts import ProtocolError
from wallx_client.protocol import decode_actions, encode_request, unpack, validate_metadata


def test_delta_model_must_return_absolute_wire_commands(cfg, metadata):
    contract = metadata["wallx_client"]
    contract.update(
        model_action_representation="b601_joint_delta_v1",
        wire_action_representation="b601_joint_absolute_v1",
        preprocessing_sha256="a" * 64,
    )
    validate_metadata(metadata, cfg.policy)
    contract["wire_action_representation"] = "b601_joint_delta_v1"
    with pytest.raises(ProtocolError, match="absolute"):
        validate_metadata(metadata, cfg.policy)


def test_native_numpy_wire_roundtrip(cfg):
    actions = np.arange(32 * 26, dtype=np.float32).reshape(1, 32, 26)
    payload = msgpack.packb(
        {"request_id": "abc", "predict_action": actions}, default=msgpack_numpy.encode
    )
    np.testing.assert_array_equal(
        decode_actions(unpack(payload), "abc", cfg.policy), actions[0, :, :7]
    )


@pytest.mark.parametrize("shape", [(7,), (32, 7), (1, 31, 26), (2, 32, 26)])
def test_wrong_action_shape_rejected(cfg, shape):
    with pytest.raises(ProtocolError):
        decode_actions({"request_id": "x", "predict_action": np.zeros(shape)}, "x", cfg.policy)


def test_nonfinite_and_wrong_request_rejected(cfg):
    bad = np.zeros((32, 26))
    bad[-1, -1] = np.nan
    with pytest.raises(ProtocolError):
        decode_actions({"request_id": "x", "predict_action": bad}, "x", cfg.policy)
    with pytest.raises(ProtocolError):
        decode_actions(
            {"request_id": "old", "predict_action": np.zeros((32, 26))}, "new", cfg.policy
        )


def test_pickle_ndarray_is_never_deserialized():
    payload = msgpack.packb(
        {b"nd": True, b"kind": b"O", b"type": "|O", b"shape": [1], b"data": b"untrusted"}
    )
    with pytest.raises(ProtocolError):
        unpack(payload)


@pytest.mark.parametrize(
    "field,value",
    [
        ("checkpoint_sha256", "0" * 64),
        ("camera_names", ["side_view", "front_view"]),
        ("action_units", "radians"),
        ("action_horizon", 16),
    ],
)
def test_server_contract_mismatch(cfg, metadata, field, value):
    metadata["wallx_client"][field] = value
    with pytest.raises(ProtocolError):
        validate_metadata(metadata, cfg.policy)


def test_rgb_and_physical_joint_payload(cfg, observation):
    request_id, packed = encode_request(observation, cfg.task, cfg.policy)
    request = msgpack.unpackb(packed, raw=False)
    assert request["request_id"] == request_id
    assert request["state"]["joint_state"] == observation.state.tolist()
    assert request["instruction"] == cfg.task
    assert request["infer_mode"] == "flow"
    jpeg = np.frombuffer(base64.b64decode(request["views"]["front_view"]), dtype=np.uint8)
    rgb = cv2.cvtColor(cv2.imdecode(jpeg, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
    np.testing.assert_allclose(rgb.mean(axis=(0, 1)), [255, 30, 15], atol=3)


def test_old_observation_rejected(observation, cfg):
    old = replace(observation, captured_monotonic=observation.captured_monotonic - 5)
    with pytest.raises(ProtocolError):
        old.validate(cfg.policy.camera_names, 0.15)
