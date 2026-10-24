"""Replay saved observations through the in-process backend, without hardware.

Produces the same report schema as inference_probe.py for compare_inference.py.
Original NPZ files and their exported requests must both match, including JPEG.
"""

import argparse
import copy
import hashlib
import importlib.metadata
import json
import os
import random
import socket
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from unittest.mock import patch


def fingerprint(value):
    import numpy as np
    import torch

    if isinstance(value, torch.Tensor):
        array = value.detach().contiguous().reshape(-1).view(torch.uint8).cpu().numpy()
        return {
            "shape": list(value.shape),
            "dtype": str(value.dtype),
            "sha256": hashlib.sha256(array.tobytes()).hexdigest(),
        }
    if isinstance(value, np.ndarray):
        return {
            "shape": list(value.shape),
            "dtype": str(value.dtype),
            "sha256": hashlib.sha256(value.tobytes()).hexdigest(),
        }
    if isinstance(value, dict):
        return {k: fingerprint(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [fingerprint(v) for v in value]
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    raise TypeError(f"Unhandled preprocessing value: {type(value)}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", type=Path, required=True)
    parser.add_argument("--machine", type=Path, required=True)
    parser.add_argument("--requests", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seeds", nargs="+", type=int, default=[7, 41])
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[2]
    sys.path[:0] = [str(root), str(root / "deployment/client/src")]
    import numpy as np
    import yaml
    from wallx_client.config import load_config
    from wallx_client.contracts import Observation

    from deployment.cli import load_deployment
    from deployment.local_policy import LocalPolicyConnection
    from deployment.local_runtime import local_deployment, native_policy

    deployment = local_deployment(load_deployment(args.profile, args.machine))
    with tempfile.TemporaryDirectory() as temp:
        path = Path(temp) / "client.yml"
        path.write_text(yaml.safe_dump(deployment["client"]))
        cfg = load_config(path)
    corpus = json.loads(args.requests.read_text())["cases"]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    log = args.output.with_suffix(".model.log")
    rows, prepared, raw, received = [], [], [], []
    owner = os.getpid()

    def forbidden(*a, **kw):
        raise AssertionError("In-process model attempted a socket or subprocess operation")

    popen = subprocess.Popen
    runtime_probes = []

    def checked_popen(command, *a, **kw):
        # Preserve native Triton discovery/JIT compilation. Record its helper
        # commands for review; the observed policy call must stay in owner PID.
        runtime_probes.append(command if isinstance(command, str) else list(map(str, command)))
        return popen(command, *a, **kw)

    with native_policy(deployment, log) as policy:
        import torch

        construct, infer = policy.model_wrapper.construct_model_input, policy.infer

        def capture(*a, **kw):
            result = construct(*a, **kw)
            prepared.append(fingerprint(result))
            return result

        def observe(body):
            assert os.getpid() == owner
            received.append(copy.deepcopy(body))
            result = infer(body)
            raw.append(np.asarray(result["predict_action"]).copy())
            return result

        policy.model_wrapper.construct_model_input, policy.infer = capture, observe
        metadata = policy.metadata
        with (
            patch.object(socket.socket, "bind", forbidden),
            patch.object(socket.socket, "connect", forbidden),
            patch.object(subprocess, "Popen", checked_popen),
            LocalPolicyConnection(cfg.policy, policy, rtc=cfg.rtc) as connection,
        ):
            for repeat in range(2):
                for case in corpus:
                    source = Path(case["source"])
                    assert hashlib.sha256(source.read_bytes()).hexdigest() == case["source_sha256"]
                    with np.load(source, allow_pickle=False) as data:
                        now = time.monotonic()
                        obs = Observation(
                            data["state"].copy(),
                            {k: data[k].copy() for k in cfg.policy.camera_names},
                            now,
                            time.time(),
                            now,
                        )
                    for seed in args.seeds:
                        policy.reset()
                        random.seed(seed)
                        np.random.seed(seed)
                        torch.manual_seed(seed)
                        torch.cuda.manual_seed_all(seed)
                        torch.cuda.synchronize()
                        before = len(prepared)
                        result = connection.predict(obs, case["task"])
                        assert len(prepared) == before + 1
                        body = received[-1]
                        body["request_id"] = case["request"]["request_id"]
                        assert body == case["request"], "Input/JPEG differs from mature C-S replay"
                        row = dict(
                            case=case["id"],
                            seed=seed,
                            repeat=repeat,
                            raw_actions=raw[-1].tolist(),
                            client_actions=result.actions.tolist(),
                            prepared=prepared[-1],
                            roundtrip_s=result.roundtrip_s,
                        )
                        rows.append(row)
                        print(
                            json.dumps(
                                {k: row[k] for k in ("case", "seed", "repeat", "roundtrip_s")}
                            ),
                            flush=True,
                        )
        assert not {"motorbridge", "can", "lerobot_robot_seeed_b601"} & sys.modules.keys()
        assets = {}
        for name in (
            "config.yml",
            "config.json",
            "norm_stats.json",
            "normalizer_action.pth",
            "normalizer_propri.pth",
            "preprocessor_config.json",
            "tokenizer.json",
            "tokenizer_config.json",
            "vocab.json",
        ):
            assets[name] = hashlib.sha256(
                (Path(deployment["checkpoint_path"]) / name).read_bytes()
            ).hexdigest()
        report = dict(
            backend="in_process",
            pid=owner,
            inference_port=None,
            socket_operations_forbidden_during_inference=True,
            native_infer_runs_in_owner_process=True,
            runtime_subprocess_commands=runtime_probes,
            source_root=str(root),
            python=sys.executable,
            gpu=torch.cuda.get_device_name(),
            metadata=metadata,
            checkpoint_assets=assets,
            requests_sha256=hashlib.sha256(args.requests.read_bytes()).hexdigest(),
            versions={
                k: importlib.metadata.version(k)
                for k in ("torch", "transformers", "numpy", "huggingface-hub", "flash-attn")
            },
            peak_allocated_bytes=torch.cuda.max_memory_allocated(),
            camera_can_motor_opened=False,
            rows=rows,
        )
    args.output.write_text(json.dumps(report, allow_nan=False), encoding="utf-8")


if __name__ == "__main__":
    main()
