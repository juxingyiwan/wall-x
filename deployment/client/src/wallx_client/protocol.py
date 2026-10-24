"""Native Wall-X MessagePack protocol with a checked joint-state extension."""

import base64
import math
import uuid

import msgpack
import numpy as np

from .config import PolicyConfig
from .contracts import ACTION_UNITS, STATE_UNITS, Observation, ProtocolError


def _numpy_hook(obj):
    """Decode msgpack-numpy numeric arrays, never its object/pickle extension."""

    def get(key, default=None):
        return obj.get(key.encode(), obj.get(key, default))

    if get("nd") is None:
        return obj
    if get("nd") is not True or get("kind", b"") not in (b"", ""):
        raise ProtocolError("Only numeric ndarray responses are accepted")
    dtype = np.dtype(get("type"))
    if dtype.kind not in "biuf" or dtype.hasobject:
        raise ProtocolError("Unsafe or unsupported response dtype")
    shape = get("shape")
    if not isinstance(shape, (list, tuple)) or not 1 <= len(shape) <= 3:
        raise ProtocolError("Invalid response array rank")
    if any(type(x) is not int or x <= 0 for x in shape) or math.prod(shape) > 1_000_000:
        raise ProtocolError("Invalid response array dimensions")
    data = get("data")
    if not isinstance(data, bytes) or len(data) != math.prod(shape) * dtype.itemsize:
        raise ProtocolError("Array bytes do not match its shape")
    return np.frombuffer(data, dtype=dtype).reshape(shape).copy()


def unpack(raw: bytes | str) -> dict:
    if isinstance(raw, str):
        raise ProtocolError(f"Wall-X server error: {raw[:2000]}")
    try:
        value = msgpack.unpackb(raw, raw=False, strict_map_key=False, object_hook=_numpy_hook)
    except ProtocolError:
        raise
    except Exception as exc:
        raise ProtocolError(f"Invalid MessagePack response: {exc}") from exc
    if not isinstance(value, dict):
        raise ProtocolError("Response must be a mapping")
    return value


def validate_metadata(metadata: dict, cfg: PolicyConfig) -> dict:
    if not isinstance(metadata, dict):
        raise ProtocolError("Policy metadata must be a mapping")
    manifest = metadata.get("wallx_client")
    if not isinstance(manifest, dict) or manifest.get("protocol_version") != 1:
        raise ProtocolError(
            "Server lacks joint_state_v1 support; see docs/rebot.md#network in the source repository"
        )
    expected = {
        "state_schema": "joint_state_v1",
        "state_units": STATE_UNITS,
        "action_units": ACTION_UNITS,
        "state_dim": 7,
        "action_dim": sum(cfg.action_dims),
        "action_horizon": cfg.horizon,
        "camera_names": list(cfg.camera_names),
        "checkpoint_id": cfg.checkpoint_id,
        "request_id_echo": True,
        "padding": cfg.padding,
    }
    if cfg.checkpoint_sha256:
        expected["checkpoint_sha256"] = cfg.checkpoint_sha256
    for key, value in expected.items():
        if manifest.get(key) != value:
            raise ProtocolError(
                f"Server {key} mismatch: expected {value!r}, got {manifest.get(key)!r}"
            )
    representation = manifest.get("model_action_representation")
    if representation is not None:
        if representation not in ("b601_joint_absolute_v1", "b601_joint_delta_v1"):
            raise ProtocolError("Unsupported B601 model action representation")
        if manifest.get("wire_action_representation") != "b601_joint_absolute_v1":
            raise ProtocolError("B601 client requires absolute command targets on the wire")
        checksum = manifest.get("preprocessing_sha256", "")
        if (
            not isinstance(checksum, str)
            or len(checksum) != 64
            or any(c not in "0123456789abcdef" for c in checksum)
        ):
            raise ProtocolError("B601 server lacks a valid preprocessing manifest checksum")
    return manifest


def encode_request(
    obs: Observation, instruction: str, cfg: PolicyConfig, rtc_request=None
) -> tuple[str, bytes]:
    request_id, body = build_request(obs, instruction, cfg, rtc_request)
    return request_id, msgpack.packb(body, use_bin_type=True)


def build_request(
    obs: Observation, instruction: str, cfg: PolicyConfig, rtc_request=None
) -> tuple[str, dict]:
    """Identical model inputs for network and in-process inference, including JPEG."""
    import cv2

    request_id = uuid.uuid4().hex
    views = {}
    for name in cfg.camera_names:
        bgr = cv2.cvtColor(obs.views[name], cv2.COLOR_RGB2BGR)
        ok, encoded = cv2.imencode(".jpg", bgr, [cv2.IMWRITE_JPEG_QUALITY, cfg.jpeg_quality])
        if not ok:
            raise ProtocolError(f"JPEG encoding failed for {name}")
        views[name] = base64.b64encode(encoded).decode("ascii")
    body = {
        "request_id": request_id,
        "state": {"joint_state": obs.state.tolist()},
        "views": views,
        "instruction": instruction,
        "infer_mode": "flow",
    }
    if rtc_request is not None:
        body["rtc"] = rtc_request
    return request_id, body


def decode_actions(response: dict, request_id: str, cfg: PolicyConfig) -> np.ndarray:
    if not isinstance(response, dict):
        raise ProtocolError("Policy response must be a mapping")
    if response.get("request_id") != request_id:
        raise ProtocolError("Response request_id mismatch")
    if "predict_action" not in response:
        raise ProtocolError("Expected predict_action; start server with --no-serialize-actions")
    try:
        actions = np.asarray(response["predict_action"])
        # Match the wire decoder; never silently coerce complex/object arrays
        # just because infer() returned them directly without MessagePack.
        if actions.dtype.kind not in "biuf" or actions.dtype.hasobject:
            raise ProtocolError("Unsafe or unsupported action dtype")
        actions = actions.astype(np.float32, copy=False)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ProtocolError("Action chunk must contain numeric values") from exc
    if actions.shape == (1, cfg.horizon, sum(cfg.action_dims)):
        actions = actions[0]
    if actions.shape != (cfg.horizon, sum(cfg.action_dims)) or not np.isfinite(actions).all():
        raise ProtocolError(f"Invalid action chunk: {actions.shape}, expected (32, 26)")
    # Server already unnormalizes; the last 19 channels are virtual padding.
    return actions[:, :7].copy()
