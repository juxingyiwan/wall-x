"""Check the native trainer's actual initialization route before RoboTwin training."""

import argparse
import json
from pathlib import Path
from types import SimpleNamespace


def audit_initialization(config_path):
    from wall_x.config.loader import load_config
    from wall_x.trainer.fsdp_trainer.fsdp_trainer import FSDPTrainer

    config_path = Path(config_path).resolve()
    cfg = load_config(str(config_path))
    if cfg.data.lerobot_config.get("adapter") != "robotwin":
        raise ValueError("This preflight is for RoboTwin configurations")
    if cfg.checkpoint.resume_model:
        raise ValueError(
            "This native trainer ignores checkpoint.resume_model. Use "
            "checkpoint.resume_from with the Wall-OSS model.safetensors file."
        )
    # Call the native decision itself, without allocating a model or CUDA.
    if not FSDPTrainer._resume_from_single_file(SimpleNamespace(cfg=cfg)):
        raise ValueError(
            "New RoboTwin fine-tuning requires a native single-file weight load"
        )
    weights = Path(cfg.checkpoint.resume_from).resolve()
    if not weights.is_file():
        raise FileNotFoundError(weights)
    return {
        "config": str(config_path),
        "initialization_route": "FSDPTrainer._resume_from_single_file",
        "weights": str(weights),
        "weights_bytes": weights.stat().st_size,
        "output": cfg.checkpoint.save_path,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    print(json.dumps(audit_initialization(args.config), indent=2))


if __name__ == "__main__":
    main()
