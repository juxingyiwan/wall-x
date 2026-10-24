import threading
import time
from dataclasses import replace

import msgpack
import msgpack_numpy
import numpy as np
import pytest
from websockets.sync.server import serve

from wallx_client.contracts import TransportError
from wallx_client.transport import WallXConnection


@pytest.mark.parametrize("delay", [0, 0.15])
def test_loopback_native_protocol_and_timeout(cfg, observation, metadata, delay):
    received = []

    def handler(ws):
        ws.send(msgpack.packb(metadata))
        request = msgpack.unpackb(ws.recv(), raw=False)
        received.append(request)
        if delay:
            time.sleep(delay)
            return
        reply = {
            "request_id": request["request_id"],
            "predict_action": np.zeros((1, 32, 26), dtype=np.float32),
        }
        ws.send(msgpack.packb(reply, default=msgpack_numpy.encode))

    with serve(handler, "127.0.0.1", 0) as server:
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        policy = replace(
            cfg.policy,
            uri=f"ws://127.0.0.1:{server.socket.getsockname()[1]}",
            request_timeout_s=0.05,
        )
        try:
            with WallXConnection(policy) as conn:
                if delay:
                    with pytest.raises(TransportError):
                        conn.predict(observation, cfg.task)
                    assert conn.socket is None
                else:
                    result = conn.predict(observation, cfg.task, generation=3)
                    assert result.actions.shape == (32, 7)
                    assert result.generation == 3
                    assert received[0]["state"]["joint_state"] == observation.state.tolist()
        finally:
            server.shutdown()
            thread.join(timeout=2)


def test_rtc_bootstrap_and_guided_request_over_websocket(cfg, observation, metadata):
    from wallx_client.config import RTCConfig
    from wallx_client.rtc import request as rtc_request

    rtc = RTCConfig(enabled=True)
    metadata = {
        **metadata,
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
    received = []

    def handler(ws):
        ws.send(msgpack.packb(metadata))
        for _ in range(2):
            payload = msgpack.unpackb(ws.recv(), raw=False)
            received.append(payload)
            req = payload["rtc"]
            ack = {
                key: req[key]
                for key in (
                    "version",
                    "epoch",
                    "plan_id",
                    "committed_steps",
                    "guidance_horizon",
                    "guidance_mode",
                )
            }
            ack["mode"] = "bootstrap" if req["prefix"] is None else "guided"
            ws.send(
                msgpack.packb(
                    dict(
                        request_id=payload["request_id"],
                        rtc=ack,
                        predict_action=np.zeros((1, 32, 26), dtype=np.float32),
                    ),
                    default=msgpack_numpy.encode,
                )
            )

    with serve(handler, "127.0.0.1", 0) as server:
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        policy = replace(cfg.policy, uri=f"ws://127.0.0.1:{server.socket.getsockname()[1]}")
        try:
            with WallXConnection(policy, rtc=rtc) as conn:
                assert conn.predict(observation, cfg.task).rtc["mode"] == "bootstrap"
                sent = rtc_request(rtc, 0, 1, np.ones((32, 7)), 14)
                result = conn.predict(observation, cfg.task, rtc_request=sent)
                assert result.rtc["mode"] == "guided"
                assert received[1]["rtc"] == sent
        finally:
            server.shutdown()
            thread.join(timeout=2)
