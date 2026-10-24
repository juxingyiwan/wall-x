"""Real-weight C-S replay on loopback; no cameras, CAN, motors or live service."""

import argparse
import asyncio
import copy
import hashlib
import importlib.metadata
import json
import os
import random
import sys
import time
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", required=True, type=Path)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--requests", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--profile", required=True, type=Path, help="Shared deployment profile")
    parser.add_argument("--seeds", nargs="+", type=int, default=[7, 41])
    args = parser.parse_args()
    import yaml

    model = yaml.safe_load(args.profile.read_text())["model"]
    adapter_root = Path(__file__).resolve().parents[2]
    sys.path[:0] = [str(args.source_root), str(adapter_root / "deployment/client/src")]
    os.environ.update(
        WALLX_SMOOTH_ACTION="false",
        WALLX_SMOOTH_GRIPPER="true",
        WALLX_VISION_ATTN_IMPLEMENTATION="flash_attention_2",
        ENABLE_FAST_PREPROCESS="false",
        ENABLE_CUDA_GRAPH="0",
        ENABLE_EXPERIMENTAL_INFERENCE_ENGINE="false",
        HF_HUB_OFFLINE="1",
        TOKENIZERS_PARALLELISM="false",
    )
    import msgpack
    import numpy as np
    import torch
    from wallx_client.config import PolicyConfig
    from wallx_client.protocol import decode_actions, unpack, validate_metadata
    from websockets.asyncio.client import connect
    from websockets.asyncio.server import serve

    from wall_x._vendor.harrix.serving.launch_serving import (
        Args,
        EnvMode,
        ServerModelConfig,
        create_policy,
    )
    from wall_x._vendor.harrix.serving.websocket_policy_server import WebsocketPolicyServer

    assert torch.cuda.is_available(), "Use an allocated CUDA device for this probe"
    policy = create_policy(
        Args(
            env=EnvMode.X2ROBOT,
            image_passing_mode="base64",
            serialize_actions=False,
            model_config=ServerModelConfig(
                checkpoint_path=str(args.checkpoint),
                train_config_path=str(args.checkpoint / "config.yml"),
                action_horizon=32,
                robot_type="desktop",
                robot_action_interpolate_multiplier=1,
                robot_action_end_ratio=1.0,
                model_device="cuda",
                norm_key=model["norm_key"],
            ),
        )
    )
    client_cfg = PolicyConfig(
        checkpoint_id=model["checkpoint_id"],
        checkpoint_sha256=model["checkpoint_sha256"],
    )
    metadata = policy.metadata
    validate_metadata(metadata, client_cfg)
    server = WebsocketPolicyServer(policy, metadata=metadata)
    prepared = []

    def fingerprint(value):
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

    construct = policy.model_wrapper.construct_model_input

    def capture(*a, **kw):
        result = construct(*a, **kw)
        prepared.append(fingerprint(result))
        return result

    # This observer returns the original input objects without altering them.
    policy.model_wrapper.construct_model_input = capture
    corpus = json.loads(args.requests.read_text())["cases"]

    async def replay():
        rows = []
        async with serve(
            server._handler, "127.0.0.1", 0, compression=None, max_size=None, ping_interval=None
        ) as listener:
            port = listener.sockets[0].getsockname()[1]
            async with connect(
                f"ws://127.0.0.1:{port}", compression=None, max_size=None, ping_interval=None
            ) as socket:
                validate_metadata(unpack(await socket.recv()), client_cfg)
                for repeat in range(2):
                    for case in corpus:
                        for seed in args.seeds:
                            policy.reset()
                            random.seed(seed)
                            np.random.seed(seed)
                            torch.manual_seed(seed)
                            torch.cuda.manual_seed_all(seed)
                            torch.cuda.synchronize()
                            before = len(prepared)
                            start = time.perf_counter()
                            await socket.send(
                                msgpack.packb(copy.deepcopy(case["request"]), use_bin_type=True)
                            )
                            response = unpack(await asyncio.wait_for(socket.recv(), timeout=180))
                            decoded = decode_actions(
                                response, case["request"]["request_id"], client_cfg
                            )
                            elapsed = time.perf_counter() - start
                            assert len(prepared) == before + 1
                            rows.append(
                                {
                                    "case": case["id"],
                                    "seed": seed,
                                    "repeat": repeat,
                                    "raw_actions": np.asarray(response["predict_action"]).tolist(),
                                    "client_actions": decoded.tolist(),
                                    "prepared": prepared[-1],
                                    "roundtrip_s": elapsed,
                                }
                            )
                            print(
                                json.dumps(
                                    {
                                        "case": case["id"],
                                        "seed": seed,
                                        "repeat": repeat,
                                        "roundtrip_s": elapsed,
                                    }
                                ),
                                flush=True,
                            )
        return rows

    rows = asyncio.run(replay())
    assets = {}
    for name in [
        "config.yml",
        "config.json",
        "norm_stats.json",
        "normalizer_action.pth",
        "normalizer_propri.pth",
        "preprocessor_config.json",
        "tokenizer.json",
        "tokenizer_config.json",
        "vocab.json",
    ]:
        assets[name] = hashlib.sha256((args.checkpoint / name).read_bytes()).hexdigest()
    report = {
        "source_root": str(args.source_root),
        "python": sys.executable,
        "gpu": torch.cuda.get_device_name(),
        "metadata": metadata,
        "checkpoint_assets": assets,
        "requests_sha256": hashlib.sha256(args.requests.read_bytes()).hexdigest(),
        "versions": {
            k: importlib.metadata.version(k)
            for k in ["torch", "transformers", "numpy", "huggingface-hub", "flash-attn"]
        },
        "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
        "camera_can_motor_opened": False,
        "rows": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, allow_nan=False), encoding="utf-8")


if __name__ == "__main__":
    main()
