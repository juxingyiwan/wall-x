"""One-process local deployment: native policy object plus the existing controller.

No model or hardware imports during inspection, no inference listener, and no
server subprocess. The client owns hardware shutdown before releasing the model.
"""

import copy
import json
import logging
import os
import signal
import sys
import tempfile
import time
import uuid
from contextlib import contextmanager
from pathlib import Path

import yaml

from deployment import cli


def local_deployment(deployment):
    effective = copy.deepcopy(deployment)
    if not effective["checkpoint_path"]:
        raise ValueError("Local inference requires checkpoint_root on this machine")
    # This is a backend label, not a network address. Server host/port are unused.
    effective["client"]["policy"]["uri"] = "local://wall-x"
    return effective


@contextmanager
def termination_signals():
    previous = {}

    def interrupted(signum, _frame):
        raise KeyboardInterrupt(signal.Signals(signum).name)

    try:
        for sig in (signal.SIGINT, signal.SIGTERM, signal.SIGHUP):
            previous[sig] = signal.signal(sig, interrupted)
        yield
    finally:
        for sig, handler in previous.items():
            signal.signal(sig, handler)


def inference_environment(deployment):
    # Share the existing C-S pins; the server command is never executed here.
    _, _, values = cli.server_command(deployment)
    keys = (
        "WALLX_SMOOTH_ACTION",
        "WALLX_SMOOTH_GRIPPER",
        "WALLX_VISION_ATTN_IMPLEMENTATION",
        "ENABLE_FAST_PREPROCESS",
        "ENABLE_CUDA_GRAPH",
        "ENABLE_EXPERIMENTAL_INFERENCE_ENGINE",
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
        "HF_HUB_OFFLINE",
        "TOKENIZERS_PARALLELISM",
    )
    return {**{key: values[key] for key in keys}, "CUDA_VISIBLE_DEVICES": deployment["cuda_id"]}


def model_options(deployment):
    checkpoint = deployment["checkpoint_path"]
    return {
        "checkpoint_path": checkpoint,
        "train_config_path": str(Path(checkpoint) / "config.yml"),
        "action_horizon": deployment["model"]["horizon"],
        "robot_type": deployment["server"]["robot_type"],
        "robot_action_interpolate_multiplier": 1,
        "robot_action_end_ratio": 1.0,
        "model_device": deployment["server"]["device"],
        "norm_key": deployment["model"]["norm_key"],
        "padding": deployment.get("padding", "zero"),
    }


def create_native_policy(deployment):
    import torch

    from wall_x._vendor.harrix.serving.launch_serving import (
        Args,
        EnvMode,
        ServerModelConfig,
        create_policy,
    )

    torch.set_num_threads(4)
    return create_policy(
        Args(
            env=EnvMode.X2ROBOT,
            image_passing_mode=deployment["model"]["image_passing_mode"],
            serialize_actions=False,
            model_config=ServerModelConfig(**model_options(deployment)),
        )
    )


@contextmanager
def native_policy(deployment, log_path):
    cli.verify_server_compatibility()
    cli.verify_checkpoint(deployment)
    values = inference_environment(deployment)
    if "torch" in sys.modules and sys.modules["torch"].cuda.is_initialized():
        raise ValueError("Start local mode in a fresh process before CUDA is initialized")
    previous = {key: os.environ.get(key) for key in values}
    log_path.parent.mkdir(parents=True, exist_ok=True)
    # Exclusive creation protects earlier diagnostics from accidental overwrite.
    with log_path.open("x", encoding="utf-8") as stream:
        logger = logging.getLogger()
        handler = logging.StreamHandler(stream)
        handler.setFormatter(logging.Formatter("%(asctime)s %(name)s %(levelname)s %(message)s"))
        level = logger.level
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
        try:
            os.environ.update(values)
            yield create_native_policy(deployment)
        finally:
            for key, value in previous.items():
                if value is None:
                    os.environ.pop(key, None)
                else:
                    os.environ[key] = value
            logger.removeHandler(handler)
            logger.setLevel(level)


def run_local(deployment, args):
    effective = local_deployment(deployment)
    client_args = list(args.client_args)
    if client_args[:1] == ["--"]:
        client_args.pop(0)
    if not client_args:
        client_args = ["check"]
    sys.path.insert(0, str(cli.ROOT / "deployment/client/src"))
    from wallx_client.cli import apply_cli_overrides, build_parser
    from wallx_client.config import (
        load_config,
        require_execution_ready,
        resolve_execution,
        validate_config,
    )

    from deployment.local_policy import LocalPolicyConnection

    with tempfile.TemporaryDirectory(prefix="wallx-local-") as directory:
        config = Path(directory) / "client.yml"
        config.write_text(yaml.safe_dump(effective["client"]), encoding="utf-8")
        cfg = load_config(config, validate=False)
        if client_args == ["check"]:
            mode = "check"
            cfg = resolve_execution(cfg)
            validate_config(cfg)
        else:
            parsed = build_parser().parse_args(["--config", str(config), *client_args])
            mode = parsed.mode
            if mode not in ("doctor", "config", "replay", "preflight", "run"):
                raise ValueError(f"Use client {mode} for operations that do not need inference")
            cfg = apply_cli_overrides(cfg, parsed)
        if mode == "run" and not args.dry_run:
            require_execution_ready(cfg)
            if not sys.stdin.isatty():
                raise ValueError("run needs an interactive on-site operator terminal")
        if args.dry_run:
            print(
                json.dumps(
                    {
                        "mode": "local",
                        "backend": "in_process",
                        "inference_port": None,
                        "model_options": model_options(effective),
                        "inference_environment": inference_environment(effective),
                        "client_args": client_args,
                        "client": cfg.to_dict(),
                        "execution_mode": cfg.execution_mode,
                        "weights_verified": False,
                        "hardware_opened": False,
                    },
                    ensure_ascii=False,
                    indent=2,
                )
            )
            return 0
        if mode in ("doctor", "config"):
            return cli.run_client(effective["client"], client_args)
        log_path = args.log or (
            Path.home()
            / ".cache/wallx-unified/local-inference"
            / f"{time.strftime('%Y%m%d-%H%M%S')}-{uuid.uuid4().hex[:8]}"
            / "model.log"
        )
        log_path = log_path.expanduser().resolve()
        print(f"In-process model log: {log_path}", flush=True)
        connection = None
        try:
            with termination_signals(), native_policy(effective, log_path) as policy:
                connection = LocalPolicyConnection(cfg.policy, policy, rtc=cfg.rtc)
                metadata = connection.open()
                report = {
                    "mode": "local",
                    "backend": "in_process",
                    "pid": os.getpid(),
                    "inference_port": None,
                    "source_root": str(cli.ROOT),
                    "python": sys.executable,
                    "metadata": metadata,
                }
                with log_path.with_suffix(".json").open("x", encoding="utf-8") as manifest:
                    json.dump(report, manifest, indent=2)
                print(
                    f"Local model ready in this process (PID {os.getpid()}, no inference port, "
                    f"padding={metadata['padding']})",
                    flush=True,
                )
                if mode == "check":
                    print(json.dumps({**report, "hardware_opened": False, "passed": True}))
                    return 0

                def factory(config, *, rtc=None):
                    if config != cfg.policy or rtc != cfg.rtc:
                        raise ValueError("Local policy configuration changed after validation")
                    return connection

                return cli.run_client(effective["client"], client_args, connection_factory=factory)
        except KeyboardInterrupt:
            return 130
        except Exception as exc:
            print(f"Local deployment failed: {exc}; model log: {log_path}", file=sys.stderr)
            return 1
        finally:
            if connection is not None:
                connection.close()
