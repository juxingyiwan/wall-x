"""Exercise the shared RTC controller through both real transport adapters."""

import threading
import time
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import msgpack
import msgpack_numpy
import numpy as np
import pytest
from wallx_client.config import RTCConfig, load_config
from wallx_client.contracts import ACTION_UNITS, STATE_UNITS, Observation
from wallx_client.rtc_queue_controller import control_loop
from wallx_client.transport import WallXConnection
from websockets.exceptions import ConnectionClosed
from websockets.sync.server import serve

from deployment.local_policy import LocalPolicyConnection


@pytest.mark.parametrize("backend", ["local", "websocket"])
@pytest.mark.parametrize("guided_latency", [0.0, 0.55, 0.82])
def test_multiple_rtc_replacements_use_same_contract(backend, guided_latency, tmp_path):
    root = Path(__file__).resolve().parents[2]
    cfg = load_config(root / "deployment/client/src/wallx_client/configs/b601.example.yml")
    cfg = replace(
        cfg,
        rtc=RTCConfig(enabled=True),
        control=replace(
            cfg.control,
            duration_s=12,
        ),
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
    received, events = [], []

    def infer(body):
        received.append(body)
        req = body["rtc"]
        if req["prefix"] is not None:
            time.sleep(guided_latency)
        ack = {
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
        ack["mode"] = "bootstrap" if req["prefix"] is None else "guided"
        actions = np.zeros((32, 26), dtype=np.float32)
        return dict(predict_action=actions, rtc=ack, request_id=body["request_id"])

    def handler(ws):
        ws.send(msgpack.packb(metadata))
        try:
            while True:
                body = msgpack.unpackb(ws.recv(), raw=False)
                ws.send(msgpack.packb(infer(body), default=msgpack_numpy.encode))
        except ConnectionClosed:
            pass

    class Hardware:
        directions = np.ones(7)
        limits = np.array([[-500.0, 500.0]] * 7)
        state = np.zeros(7)
        closed = False

        def observe(self):
            now = time.monotonic()
            return Observation(
                self.state.copy(),
                {n: np.zeros((48, 64, 3), dtype=np.uint8) for n in cfg.policy.camera_names},
                now,
                time.time(),
                now,
            ), {}

        def hold_command(self, state):
            return state.copy()

        def send(self, raw):
            self.state = raw.copy()
            return {"driver_target_physical": raw.tolist()}

        def close(self):
            self.closed = True

    recorder = SimpleNamespace(
        path=tmp_path,
        event=lambda kind, **kw: events.append((kind, kw)),
        prediction_async=lambda *a: None,
        check_media=lambda: None,
        sample=lambda *a, **kw: None,
        append=lambda *a, **kw: None,
    )
    hardware = Hardware()

    def run(connection):
        connection.open()

        def stopped():
            return sum(k == "rtc_queue_replaced" and v["guided"] for k, v in events) >= 3

        return control_loop(
            hardware,
            connection,
            cfg,
            recorder,
            lambda: "q" if stopped() else "",
        )

    if backend == "local":
        policy = SimpleNamespace(metadata=metadata, infer=infer)
        report = run(LocalPolicyConnection(cfg.policy, policy, rtc=cfg.rtc))
    else:
        with serve(handler, "127.0.0.1", 0) as server:
            worker = threading.Thread(target=server.serve_forever, daemon=True)
            worker.start()
            config = replace(cfg.policy, uri=f"ws://127.0.0.1:{server.socket.getsockname()[1]}")
            try:
                report = run(WallXConnection(config, rtc=cfg.rtc))
            finally:
                server.shutdown()
                worker.join(timeout=2)
    assert report["outcome"] == "aborted", (report, events)
    assert hardware.closed
    assert report["replacements"] == 3
    assert report["target_processing"] == "raw_action_queue"
    assert received[0]["rtc"]["prefix"] is None
    assert report["guided_queries"] >= 3
    for body in received[1:]:
        req = body["rtc"]
        if req["prefix"] is not None:
            assert np.asarray(req["prefix"]).shape == (req["guidance_horizon"], 7)
            assert 1 <= req["committed_steps"] < req["guidance_horizon"] < 32
