"""Synchronous chunk semantics through both real policy adapters, without motors."""

import json
import threading
import time
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import msgpack
import msgpack_numpy
import numpy as np
import pytest
from wallx_client.config import load_config
from wallx_client.contracts import ACTION_UNITS, STATE_UNITS, Observation
from wallx_client.controller import control_loop
from wallx_client.recording import Recorder
from wallx_client.transport import WallXConnection
from websockets.exceptions import ConnectionClosed
from websockets.sync.server import serve

from deployment.local_policy import LocalPolicyConnection


@pytest.mark.parametrize("backend", ["local", "websocket"])
@pytest.mark.parametrize("slow_recording", [False, True])
def test_sync_uses_current_observation_without_extra_gates(backend, slow_recording, tmp_path):
    root = Path(__file__).resolve().parents[2]
    cfg = load_config(root / "deployment/client/src/wallx_client/configs/b601.example.yml")
    cfg = replace(
        cfg,
        output_root=str(tmp_path / "rollouts"),
        control=replace(
            cfg.control,
            execution_horizon=2,
            duration_s=5,
            max_tick_s=1,
            # These RTC knobs must not block or reject synchronous requests.
            max_response_age_s=0.001,
            max_state_drift_arm_deg=0.001,
            max_state_drift_gripper_deg=0.001,
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
        )
    }
    received, starts, completions = [], [], []
    recorder = Recorder(cfg, "sync_backends")
    release_recording, recording_started = threading.Event(), threading.Event()
    if slow_recording:
        original_prediction = recorder.prediction

        def blocked_prediction(result):
            recording_started.set()
            assert release_recording.wait(8)
            original_prediction(result)

        recorder.prediction = blocked_prediction
    original_event = recorder.event

    def event(kind, **kwargs):
        if kind == "chunk_completed":
            completions.append(time.monotonic())
        original_event(kind, **kwargs)

    recorder.event = event

    class Hardware:
        directions = np.ones(7)
        ticks = 0
        closed = False

        def observe(self):
            self.ticks += 1
            now = time.monotonic()
            return Observation(
                np.full(7, self.ticks * 3.0),
                {name: np.zeros((48, 64, 3), dtype=np.uint8) for name in cfg.policy.camera_names},
                now - 0.12,
                time.time(),
                now,
            ), {}

        def hold_command(self, state):
            return state.copy()

        def send(self, raw):
            assert not self.closed
            return {"driver_target_physical": raw.tolist()}

        def close(self):
            self.closed = True

    hardware = Hardware()

    def infer(body):
        received.append(body)
        starts.append(time.monotonic())
        # Let the current motor state change during inference. That is not a
        # reason to retry a serial request after its contract has been checked.
        time.sleep(0.05)
        actions = np.zeros((32, 26), dtype=np.float32)
        actions[:, 0] = len(received) * 10 + np.arange(32) / 100
        return dict(predict_action=actions, request_id=body["request_id"])

    def run(connection):
        connection.open()
        return control_loop(
            hardware, connection, cfg, recorder, lambda: "q" if len(completions) == 2 else ""
        )

    def handler(ws):
        ws.send(msgpack.packb(metadata))
        try:
            while True:
                body = msgpack.unpackb(ws.recv(), raw=False)
                ws.send(msgpack.packb(infer(body), default=msgpack_numpy.encode))
        except ConnectionClosed:
            pass

    report = {}
    try:
        if backend == "local":
            report = run(
                LocalPolicyConnection(cfg.policy, SimpleNamespace(metadata=metadata, infer=infer))
            )
        else:
            with serve(handler, "127.0.0.1", 0) as server:
                worker = threading.Thread(target=server.serve_forever, daemon=True)
                worker.start()
                try:
                    config = replace(
                        cfg.policy, uri=f"ws://127.0.0.1:{server.socket.getsockname()[1]}"
                    )
                    report = run(WallXConnection(config))
                finally:
                    server.shutdown()
                    worker.join(timeout=2)
        assert report["outcome"] == "aborted", report
        assert hardware.closed
        assert len(completions) == len(received) == report["queries"] == 2
        assert report["discarded_responses"] == 0
        assert starts[1] > completions[0]  # No prefetch or execution/inference overlap.
        if slow_recording:
            assert recording_started.is_set()
            assert not list(recorder.path.glob("query_*.json"))
    finally:
        release_recording.set()
        recorder.close(report)

    queries = sorted(
        [json.loads(path.read_text()) for path in recorder.path.glob("query_*.json")],
        key=lambda query: query["state_monotonic"],
    )
    assert len(queries) == len(list(recorder.path.glob("query_*.npz"))) == 2
    assert queries[1]["state_monotonic"] > completions[0]
    assert queries[1]["captured_monotonic"] < completions[0]
    commands = [
        json.loads(line) for line in (recorder.path / "commands.jsonl").read_text().splitlines()
    ]
    policy_commands = [
        row["raw_recorded_command"][0] for row in commands if row["phase"] == "policy"
    ]
    np.testing.assert_allclose(policy_commands, [10, 10.01, 20, 20.01])
    assert all("rtc" not in body for body in received)
