"""The same contract through native infer() and a real loopback WebSocket."""

import copy
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import msgpack
import msgpack_numpy
import numpy as np
import pytest
from wallx_client.config import RTCConfig, load_config
from wallx_client.contracts import (
    ACTION_UNITS,
    STATE_UNITS,
    Observation,
    ProtocolError,
    TransportError,
)
from wallx_client.protocol import encode_request, unpack
from wallx_client.rtc import request as rtc_request
from wallx_client.transport import WallXConnection
from websockets.exceptions import ConnectionClosed
from websockets.sync.server import serve

from deployment.local_policy import LocalPolicyConnection


@pytest.fixture
def case():
    root = Path(__file__).resolve().parents[2]
    cfg = load_config(root / "deployment/client/src/wallx_client/configs/b601.example.yml")
    now = time.monotonic()
    rng = np.random.default_rng(17)
    obs = Observation(
        np.array([1.5, -2.0, 3.0, -4.5, 5.0, 6.0, 42.0]),
        {
            name: rng.integers(0, 256, (48, 64, 3), dtype=np.uint8)
            for name in cfg.policy.camera_names
        },
        now,
        time.time(),
        now,
        capture_sequence=7,
        camera_generation=3,
    )
    metadata = {
        "wallx_client": dict(
            protocol_version=1,
            state_schema="joint_state_v1",
            state_units=STATE_UNITS,
            action_units=ACTION_UNITS,
            state_dim=7,
            action_dim=26,
            action_horizon=32,
            camera_names=list(cfg.policy.camera_names),
            checkpoint_id=cfg.policy.checkpoint_id,
            checkpoint_sha256=cfg.policy.checkpoint_sha256,
            request_id_echo=True,
            padding=cfg.policy.padding,
        ),
        "rtc": dict(
            version=1,
            supported=True,
            guidance="endpoint_vjp",
            guidance_modes=["lerobot", "pi_vjp"],
            action_horizon=32,
            real_action_dim=7,
            model_dt_s=1 / 30,
            prefix_units="recorded_commands",
        ),
    }
    raw = rng.normal(size=(1, 32, 26)).astype(np.float32)
    received = []

    def response(body):
        result = {
            "predict_action": raw,
            "request_id": body["request_id"],
            "server_timing": {"infer_ms": 12.0},
        }
        if "rtc" in body:
            req = body["rtc"]
            result["rtc"] = {
                k: req[k]
                for k in (
                    "version",
                    "epoch",
                    "plan_id",
                    "committed_steps",
                    "guidance_horizon",
                    "guidance_mode",
                )
            }
            result["rtc"]["mode"] = "bootstrap" if req["prefix"] is None else "guided"
        return result

    policy = SimpleNamespace(metadata=metadata, response=response)

    def infer(body):
        received.append(copy.deepcopy(body))
        return policy.response(body)

    policy.infer = infer
    return SimpleNamespace(cfg=cfg, obs=obs, policy=policy, raw=raw, received=received)


@pytest.fixture(params=["local", "websocket"])
def backend(request, case):
    @contextmanager
    def connection(rtc=None, timeout=2.0):
        config = replace(case.cfg.policy, request_timeout_s=timeout)
        if request.param == "local":
            with LocalPolicyConnection(config, case.policy, rtc=rtc) as conn:
                yield conn
            return

        def handler(ws):
            try:
                ws.send(msgpack.packb(case.policy.metadata))
                while True:
                    body = msgpack.unpackb(ws.recv(), raw=False)
                    reply = case.policy.infer(body)
                    ws.send(msgpack.packb(reply, default=msgpack_numpy.encode))
            except ConnectionClosed:
                pass

        with serve(handler, "127.0.0.1", 0) as server:
            worker = threading.Thread(target=server.serve_forever, daemon=True)
            worker.start()
            config = replace(config, uri=f"ws://127.0.0.1:{server.socket.getsockname()[1]}")
            try:
                with WallXConnection(config, rtc=rtc) as conn:
                    yield conn
            finally:
                server.shutdown()
                worker.join(timeout=2)

    return connection


@pytest.mark.parametrize("mode", ["ordinary", "bootstrap", "guided"])
def test_inputs_actions_and_observation_identity_match_existing_wire_contract(case, backend, mode):
    rtc = RTCConfig(enabled=mode != "ordinary")
    sent = rtc_request(rtc, 2, 9, np.ones((32, 7)), 14) if mode == "guided" else None
    before = copy.deepcopy(sent)
    with backend(rtc) as conn:
        result = conn.predict(case.obs, "put the pen in the drawer", 2, rtc_request=sent)
    assert sent == before
    body = case.received[0]
    expected_rtc = (
        sent if mode == "guided" else (rtc_request(rtc, 2, 0) if mode == "bootstrap" else None)
    )
    _, wire = encode_request(case.obs, "put the pen in the drawer", case.cfg.policy, expected_rtc)
    expected = unpack(wire)
    expected["request_id"] = body["request_id"]
    assert body == expected
    np.testing.assert_array_equal(result.actions, case.raw[0, :, :7])
    assert result.actions.dtype == np.float32
    assert result.observation is case.obs and result.generation == 2
    assert result.server_timing == {"infer_ms": 12.0}
    assert (result.rtc is None) == (mode == "ordinary")
    assert result.timing.call_s == result.roundtrip_s
    assert result.timing.preparation_s >= 0
    assert result.observation_to_result_s >= result.roundtrip_s
    # Do not introduce trajectory correction, sign conversion, or gripper scaling.
    case.raw[:] = 99
    assert not np.all(result.actions == 99)


@pytest.mark.parametrize(
    "kind",
    [
        "request_id",
        "shape",
        "nan_padding",
        "nonnumeric",
        "mapping",
        "timing",
        "rtc",
        "complex",
        "object",
        "string",
    ],
)
def test_invalid_results_rejected_by_both_backends(case, backend, kind):
    original = case.policy.response

    def bad(body):
        value = original(body)
        if kind == "mapping":
            return []
        if kind == "request_id":
            value["request_id"] = "stale-request"
        elif kind == "shape":
            value["predict_action"] = np.zeros((32, 7))
        elif kind == "nan_padding":
            value["predict_action"][0, -1, -1] = np.nan
        elif kind == "nonnumeric":
            value["predict_action"] = [["invalid"]]
        elif kind in {"complex", "object", "string"}:
            dtype = {"complex": np.complex64, "object": object, "string": str}[kind]
            value["predict_action"] = value["predict_action"].astype(dtype)
        elif kind == "timing":
            value["server_timing"] = []
        elif kind == "rtc":
            value["rtc"] = {"mode": "guided"}
        return value

    case.policy.response = bad
    with backend() as conn, pytest.raises(ProtocolError):
        conn.predict(case.obs, "task")


@pytest.mark.parametrize(
    "kind",
    [
        "state",
        "camera",
        "rgb",
        "empty",
        "generation",
        "rtc_disabled",
        "epoch",
        "prefix",
        "fields",
        "guidance_mode",
    ],
)
def test_invalid_inputs_fail_before_inference(case, backend, kind):
    rtc = RTCConfig(enabled=kind in {"epoch", "prefix", "fields", "guidance_mode"})
    obs, generation, sent = case.obs, 2, None
    if kind == "state":
        obs = replace(obs, state=np.full(7, np.nan))
    elif kind == "camera":
        obs = replace(obs, views={})
    elif kind in {"rgb", "empty"}:
        views = dict(obs.views)
        views[case.cfg.policy.camera_names[0]] = (
            np.zeros((48, 64), dtype=np.uint8)
            if kind == "rgb"
            else np.zeros((0, 64, 3), dtype=np.uint8)
        )
        obs = replace(obs, views=views)
    elif kind == "generation":
        generation = -1
    else:
        sent = rtc_request(rtc, 1 if kind == "epoch" else 2, 0)
        if kind == "prefix":
            sent = rtc_request(rtc, 2, 1, np.zeros((32, 7)), 14)
            sent["prefix"] = np.zeros((31, 7)).tolist()
        elif kind == "fields":
            del sent["plan_id"]
        elif kind == "guidance_mode":
            sent["guidance_mode"] = "pi_vjp"
    with backend(rtc) as conn, pytest.raises(ProtocolError):
        conn.predict(obs, "task", generation, rtc_request=sent)
    assert case.received == []


@pytest.mark.parametrize("kind", ["checkpoint_id", "action_units", "camera_names", "rtc_timebase"])
def test_handshake_rejects_mismatched_contract(case, backend, kind):
    rtc = RTCConfig(enabled=kind == "rtc_timebase")
    if kind == "rtc_timebase":
        case.policy.metadata["rtc"]["model_dt_s"] = 0.1
    elif kind == "camera_names":
        case.policy.metadata["wallx_client"][kind].reverse()
    else:
        case.policy.metadata["wallx_client"][kind] = "wrong"
    with pytest.raises(ProtocolError), backend(rtc):
        pytest.fail("Bad contract opened successfully")
    assert case.received == []


def test_mutating_native_request_cannot_rewrite_rtc_ack_expectation(case, backend):
    rtc = RTCConfig(enabled=True)
    sent = rtc_request(rtc, 2, 9, np.ones((32, 7)), 14)
    original = case.policy.response

    def tampered(body):
        body["rtc"]["plan_id"] += 1
        return original(body)

    case.policy.response = tampered
    with backend(rtc) as conn, pytest.raises(ProtocolError, match="acknowledgement"):
        conn.predict(case.obs, "task", 2, rtc_request=sent)
    assert sent["plan_id"] == 9


def test_same_monotonic_timing_and_no_invented_model_compute_time(case, backend, monkeypatch):
    original = case.policy.response

    def without_timing(body):
        value = original(body)
        del value["server_timing"]
        return value

    case.policy.response = without_timing
    clock = iter([15.0, 15.2, 15.7])
    monkeypatch.setattr("wallx_client.policy.time", SimpleNamespace(monotonic=lambda: next(clock)))
    obs = replace(case.obs, captured_monotonic=14.0, state_monotonic=14.0)
    with backend() as conn:
        result = conn.predict(obs, "task")
    assert result.timing.preparation_s == pytest.approx(0.2)
    assert result.roundtrip_s == pytest.approx(0.5)
    assert result.observation_to_result_s == pytest.approx(1.7)
    assert result.received_monotonic == result.timing.completed_monotonic == 15.7
    assert result.server_timing == {}


def test_only_one_request_and_closed_session_cannot_deliver_pending_result(case, backend):
    entered, release = threading.Event(), threading.Event()
    original = case.policy.response

    def blocked(body):
        entered.set()
        assert release.wait(3)
        return original(body)

    case.policy.response = blocked
    with backend() as conn, ThreadPoolExecutor(1) as executor:
        pending = executor.submit(conn.predict, case.obs, "task")
        try:
            assert entered.wait(2)
            with pytest.raises(ProtocolError, match="one in-flight"):
                conn.predict(case.obs, "task")
            conn.close()
        finally:
            release.set()
        with pytest.raises(TransportError):
            pending.result(timeout=3)


def test_native_missing_request_envelope_is_filled_but_never_overwritten(case):
    original = case.policy.response

    def no_envelope(body):
        value = original(body)
        del value["request_id"]
        return value

    case.policy.response = no_envelope
    with LocalPolicyConnection(case.cfg.policy, case.policy) as conn:
        result = conn.predict(case.obs, "task")
    assert result.request_id == case.received[0]["request_id"]


@pytest.mark.parametrize("prefix_steps", [20, 32])
def test_numpy_prefix_is_canonicalized_without_changing_values(case, backend, prefix_steps):
    rtc = RTCConfig(enabled=True)
    prefix = np.random.default_rng(31).normal(size=(prefix_steps, 7))
    sent = rtc_request(rtc, 2, 1, prefix, 14, guidance_horizon=20)
    sent["prefix"] = prefix
    with backend(rtc) as conn:
        result = conn.predict(case.obs, "task", 2, rtc_request=sent)
    assert result.rtc["mode"] == "guided"
    assert sent["prefix"] is prefix
    assert case.received[0]["rtc"]["prefix"] == prefix.tolist()


def test_malformed_capability_has_same_protocol_failure(case, backend):
    case.policy.metadata["rtc"]["model_dt_s"] = "invalid"
    with pytest.raises(ProtocolError, match="Malformed RTC"), backend(RTCConfig(enabled=True)):
        pytest.fail("Malformed capability accepted")


@pytest.mark.parametrize("dtype", [np.complex64, object])
def test_unsupported_prefix_dtype_is_rejected_before_either_backend(case, backend, dtype):
    rtc = RTCConfig(enabled=True)
    sent = rtc_request(rtc, 2, 1, np.ones((32, 7)), 14)
    sent["prefix"] = np.ones((32, 7), dtype=dtype)
    with backend(rtc) as conn, pytest.raises(ProtocolError, match="prefix dtype"):
        conn.predict(case.obs, "task", 2, rtc_request=sent)
    assert case.received == []
