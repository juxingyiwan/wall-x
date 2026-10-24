"""Unified deployment entry; configuration and dry runs never open hardware."""

import argparse
import copy
import json
import os
import re
import sys
import tempfile
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "deployment/configs/b601.example.yml"
LOCAL_CONFIG = ROOT / "deployment/local/profile.yml"


def mapping(value, allowed, label):
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be a mapping")
    if unknown := value.keys() - allowed:
        raise ValueError(f"Unknown {label} keys: {sorted(unknown)}")
    return value


def read_yaml(path):
    with Path(path).open(encoding="utf-8") as stream:
        return yaml.safe_load(stream)


def expand_path(value):
    if not isinstance(value, str) or not value:
        raise ValueError("Machine paths must be nonempty strings")
    expanded = os.path.expandvars(os.path.expanduser(value))
    if "$" in expanded or not Path(expanded).is_absolute():
        raise ValueError(f"Path must expand to an absolute path: {value}")
    return expanded


def merge(base, override):
    result = copy.deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = merge(result[key], value)
        else:
            result[key] = copy.deepcopy(value)
    return result


def positive_int(value):
    return type(value) is int and value > 0


def load_deployment(config=DEFAULT_CONFIG, machine=None):
    path = Path(config).resolve()
    raw = mapping(
        read_yaml(path),
        {"schema_version", "model", "client_config", "server", "padding"},
        "deployment",
    )
    if type(raw.get("schema_version")) is not int or raw["schema_version"] != 1:
        raise ValueError("Unsupported deployment schema_version")
    padding = raw.get("padding", "zero")
    if not isinstance(padding, str) or padding not in ("zero", "original"):
        raise ValueError("padding must be 'zero' or 'original'; use padding: zero")
    # Command-scoped override shared by local inference, serving and client checks.
    # Read at load time; never write the profile or change a running model.
    if "WALLX_PADDING" in os.environ:
        padding = os.environ["WALLX_PADDING"]
        if padding not in ("zero", "original"):
            raise ValueError("WALLX_PADDING must be 'zero' or 'original'")
    model_keys = {
        "checkpoint_id",
        "checkpoint_sha256",
        "epoch",
        "step",
        "norm_key",
        "horizon",
        "action_dims",
        "camera_names",
        "image_passing_mode",
    }
    model = mapping(raw.get("model"), model_keys, "model")
    if model.keys() != model_keys:
        raise ValueError("Model identity and protocol fields must all be explicit")
    # Training checkpoints use zero-based epoch numbers.
    if type(model["epoch"]) is not int or model["epoch"] < 0:
        raise ValueError("model.epoch must be a nonnegative integer")
    for key in ("step", "horizon"):
        if not positive_int(model[key]):
            raise ValueError(f"model.{key} must be a positive integer")
    identity = model["checkpoint_id"]
    if (
        not isinstance(identity, str)
        or len(identity.split("/")) != 2
        or any(p in ("", ".", "..") for p in identity.split("/"))
        or "\\" in identity
    ):
        raise ValueError("checkpoint_id must be experiment/epoch_step")
    if identity.split("/")[-1] != f"{model['epoch']}_{model['step']}":
        raise ValueError("checkpoint_id disagrees with epoch/step")
    if not isinstance(model["checkpoint_sha256"], str) or not re.fullmatch(
        "[0-9a-f]{64}", model["checkpoint_sha256"]
    ):
        raise ValueError("checkpoint_sha256 must be a lowercase SHA256 digest")
    if not isinstance(model["norm_key"], str) or not model["norm_key"]:
        raise ValueError("norm_key must be a nonempty string")
    if (
        model["horizon"] != 32
        or model["image_passing_mode"] != "base64"
        or model["action_dims"] != [6, 1, 19]
        or model["camera_names"] != ["front_view", "side_view"]
    ):
        raise ValueError("This adapter requires the base64 / joint_state_v1 protocol")
    server_keys = {"host", "port", "device", "robot_type", "env"}
    server = mapping(raw.get("server"), server_keys, "server")
    if server.keys() != server_keys:
        raise ValueError("All server fields must be explicit")
    machine_cfg = (
        {}
        if machine is None
        else mapping(
            read_yaml(machine), {"checkpoint_root", "cuda_id", "server", "client"}, "machine"
        )
    )
    server = merge(
        server, mapping(machine_cfg.get("server", {}), {"host", "port"}, "machine.server")
    )
    if (
        not isinstance(server["host"], str)
        or not server["host"]
        or not positive_int(server["port"])
        or server["port"] > 65535
    ):
        raise ValueError("Invalid server host/port")
    if (server["env"], server["robot_type"], server["device"]) != ("X2ROBOT", "desktop", "cuda"):
        raise ValueError("This adapter requires X2ROBOT / desktop / cuda serving")
    client_ref = raw.get("client_config")
    if not isinstance(client_ref, str) or not client_ref:
        raise ValueError("client_config must name a YAML file relative to this profile")
    client_keys = {
        "task",
        "output_root",
        "policy",
        "robot",
        "control",
        "display_data",
        "save_rollout",
        "display_ip",
        "display_port",
        "rtc",
        "shutdown",
    }
    client = mapping(read_yaml(path.parent / client_ref), client_keys, "client")
    if not {"task", "output_root", "policy", "robot", "control"} <= client.keys():
        raise ValueError("Client base requires task, output_root, policy, robot and control")
    if not isinstance(client["policy"], dict) or not isinstance(client["robot"], dict):
        raise ValueError("Client policy and robot must be mappings")
    overrides = mapping(
        machine_cfg.get("client", {}),
        {
            "output_root",
            "robot",
            "policy",
            "display_data",
            "save_rollout",
            "display_ip",
            "display_port",
        },
        "machine.client",
    )
    mapping(overrides.get("policy", {}), {"uri"}, "machine.client.policy")
    client = merge(client, overrides)
    shared = {
        k: model[k]
        for k in ("checkpoint_id", "checkpoint_sha256", "horizon", "action_dims", "camera_names")
    }
    shared["padding"] = padding
    policy = client.setdefault("policy", {})
    for key, value in shared.items():
        if key in policy and policy[key] != value:
            raise ValueError(f"client.policy.{key} disagrees with the shared model")
        policy[key] = value
    client["output_root"] = expand_path(client["output_root"])
    robot = client["robot"]
    robot["lock_paths"] = [expand_path(p) for p in robot.get("lock_paths", [])]
    robot["camera_devices"] = {
        k: expand_path(v) for k, v in robot.get("camera_devices", {}).items()
    }
    checkpoint = None
    if "checkpoint_root" in machine_cfg:
        checkpoint = str(Path(expand_path(machine_cfg["checkpoint_root"])) / identity)
    allocated = os.environ.get("CUDA_VISIBLE_DEVICES")
    if os.environ.get("SLURM_JOB_ID") and (allocated is None or not allocated.strip()):
        raise ValueError(
            "SLURM_JOB_ID is set but CUDA_VISIBLE_DEVICES is missing or empty. "
            "In the compute-node shell, set CUDA_VISIBLE_DEVICES to the GPU(s) "
            "verified as assigned to this allocation, then retry. "
            "Activating conda alone does not set this variable. "
            "Do not guess GPU IDs or bypass this check with cuda_id."
        )
    cuda = machine_cfg.get("cuda_id", os.environ.get("CUDA_VISIBLE_DEVICES", "0"))
    if not isinstance(cuda, str) or not cuda.strip():
        raise ValueError("cuda_id must be a nonempty string")
    if os.environ.get("SLURM_JOB_ID") and cuda != allocated:
        raise ValueError("Preserve Slurm's CUDA_VISIBLE_DEVICES; remove the cuda_id override")
    return {
        "padding": padding,
        "model": model,
        "server": server,
        "client": client,
        "checkpoint_path": checkpoint,
        "cuda_id": cuda,
        "machine": str(Path(machine).resolve()) if machine else None,
    }


def server_command(deployment):
    checkpoint = deployment["checkpoint_path"]
    if checkpoint is None:
        raise ValueError("Server mode requires --machine with checkpoint_root")
    server, model = deployment["server"], deployment["model"]
    env = os.environ.copy()
    # Pin inherited shell defaults as well as CLI arguments to the deployed behavior.
    values = {
        "PYTHON_BIN": sys.executable,
        "CHECKPOINT_PATH": checkpoint,
        "TRAIN_CONFIG_PATH": str(Path(checkpoint) / "config.yml"),
        "HOST": server["host"],
        "PORT": str(server["port"]),
        "WALLX_ENV": server["env"],
        "CUDA_ID": deployment["cuda_id"],
        "ROBOT_TYPE": server["robot_type"],
        "MODEL_DEVICE": server["device"],
        "ACTION_HORIZON": str(model["horizon"]),
        "IMAGE_PASSING_MODE": model["image_passing_mode"],
        "SERIALIZE_ACTIONS": "0",
        "ROBOT_ACTION_INTERPOLATE_MULTIPLIER": "1",
        "ROBOT_ACTION_END_RATIO": "1.0",
        "WALLX_SMOOTH_ACTION": "false",
        "WALLX_SMOOTH_GRIPPER": "true",
        "WALLX_VISION_ATTN_IMPLEMENTATION": "flash_attention_2",
        "ENABLE_FAST_PREPROCESS": "false",
        "ENABLE_CUDA_GRAPH": "0",
        "ENABLE_EXPERIMENTAL_ENGINE": "0",
        "ENABLE_EXPERIMENTAL_INFERENCE_ENGINE": "false",
        "MAX_BATCH_SIZE": "",
        "DEFAULT_PROMPT": "",
        "DEBUG": "0",
        "OMP_NUM_THREADS": "4",
        "MKL_NUM_THREADS": "4",
        "HF_HUB_OFFLINE": "1",
        "TOKENIZERS_PARALLELISM": "false",
        "PYTHONUNBUFFERED": "1",
    }
    env.update(values)
    # The native script prepends ROOT again. Neither location points at the old checkout.
    env["PYTHONPATH"] = str(ROOT)
    command = [
        "bash",
        str(ROOT / "scripts/run_serving.sh"),
        "--",
        "--model-config.norm-key",
        model["norm_key"],
        "--model-config.padding",
        deployment.get("padding", "zero"),
    ]
    return command, env, values


def verify_server_compatibility():
    required = [
        ROOT / "wall_x/_vendor/harrix/serving/_wallx_infer/joint_state.py",
    ]
    if any(not path.is_file() for path in required):
        raise ValueError(
            "This server lacks the joint_state_v1 prerequisite; see docs/rebot.md#network. "
            "Configuration inspection and --dry-run do not establish server compatibility."
        )


def verify_checkpoint(deployment):
    if deployment["model"]["checkpoint_sha256"] == "0" * 64:
        raise ValueError("Replace the example checkpoint SHA256 before serving")
    checkpoint = Path(deployment["checkpoint_path"])
    if not (checkpoint / "config.yml").is_file():
        raise ValueError(f"Missing training configuration: {checkpoint / 'config.yml'}")
    sys.path.insert(0, str(ROOT))
    from wall_x.checkpoint_weights import checkpoint_sha256, resolve_checkpoint_weights

    weights = resolve_checkpoint_weights(checkpoint)
    if checkpoint_sha256(weights) != deployment["model"]["checkpoint_sha256"]:
        raise ValueError(f"Checkpoint SHA256 differs from the shared model pin: {weights}")


def run_client(config, argv, *, connection_factory=None):
    # Explicit source selection avoids using the old editable installation.
    sys.path.insert(0, str(ROOT / "deployment/client/src"))
    from wallx_client.cli import main as client_main

    with tempfile.TemporaryDirectory(prefix="wallx-unified-") as tmp:
        path = Path(tmp) / "client.yml"
        path.write_text(yaml.safe_dump(config), encoding="utf-8")
        options = {} if connection_factory is None else {"connection_factory": connection_factory}
        return client_main(["--config", str(path), *argv], **options)


def main(argv=None):
    parser = argparse.ArgumentParser(description="Wall-X B601 deployment adapter")
    parser.add_argument(
        "--config", type=Path, default=LOCAL_CONFIG if LOCAL_CONFIG.is_file() else DEFAULT_CONFIG
    )
    parser.add_argument("--machine", type=Path, help="Machine override YAML; never auto-selected")
    sub = parser.add_subparsers(dest="mode", required=True)
    sub.add_parser("config", help="Print shared deployment configuration; no hardware access")
    server = sub.add_parser("server", help="Run the existing native inference server")
    server.add_argument(
        "--dry-run", action="store_true", help="Print command/env without loading weights"
    )
    sub.add_parser("client", add_help=False, help="Forward to the imported client CLI")
    local = sub.add_parser(
        "local", help="Call the native model inside the robot control process; no inference port"
    )
    local.add_argument("--log", type=Path, help="New local model log file; never overwritten")
    local.add_argument("--dry-run", action="store_true", help="Inspect local commands only")
    local.add_argument(
        "client_args",
        nargs=argparse.REMAINDER,
        help="check (default), or -- followed by existing client options and replay/preflight/run",
    )
    # Client arguments retain their original ordering and parser, including --help.
    args, remaining = parser.parse_known_args(argv)
    if args.mode != "client" and remaining:
        parser.error(f"unrecognized arguments: {' '.join(remaining)}")
    forwarded = args.client_args if args.mode == "local" else remaining
    if args.mode in ("client", "local") and any(
        v.startswith("--")
        and len(v.partition("=")[0]) >= 3
        and "--config".startswith(v.partition("=")[0])
        for v in forwarded
    ):
        parser.error("Use the shared top-level --config; client config overrides are disabled")
    try:
        deployment = load_deployment(args.config, args.machine)
        if args.mode == "local":
            if args.machine is None:
                raise ValueError(
                    "Local mode requires --machine with checkpoint_root and robot settings"
                )
            sys.path.insert(0, str(ROOT))
            from deployment.local_runtime import run_local

            return run_local(deployment, args)
        if args.mode == "config":
            print(json.dumps(deployment, ensure_ascii=False, indent=2))
            return 0
        if args.mode == "client":
            if args.machine is None and not any(v in ("-h", "--help") for v in remaining):
                raise ValueError(
                    "Client mode requires --machine with explicit cameras and shared locks"
                )
            return run_client(deployment["client"], remaining)
        command, env, values = server_command(deployment)
        if args.dry_run:
            print(
                json.dumps(
                    {
                        "command": command,
                        "environment": values,
                        "cwd": str(ROOT),
                        "checkpoint_sha256": deployment["model"]["checkpoint_sha256"],
                        "weights_verified": False,
                        "hardware_opened": False,
                    },
                    indent=2,
                )
            )
            return 0
        verify_server_compatibility()
        verify_checkpoint(deployment)
        os.chdir(ROOT)
        os.execvpe(command[0], command, env)
    except (OSError, ValueError, yaml.YAMLError) as exc:
        print(f"wallx: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
