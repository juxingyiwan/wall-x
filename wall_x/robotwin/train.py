#!/usr/bin/env python3
"""
FSDP Training Entry Point
Launch with torchrun:
    torchrun --nproc_per_node=8 --master_port=29500 train_fsdp.py --config config.yaml
"""
import argparse
import dataclasses
import hashlib
import json
import logging
import os
import time
from pathlib import Path

import torch
import wandb

from wall_x.robotwin.config import load_config as load_typed_config
from wall_x.robotwin.logging_utils import training_logging
from wall_x.robotwin.video import install_bounded_video_reader
from wall_x.trainer.fsdp_trainer import (
    cleanup_distributed,
    is_main_process,
)

logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description="FSDP Training Script")
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to training config YAML file",
    )
    # FSDP specific overrides
    parser.add_argument(
        "--fsdp_sharding_strategy",
        type=str,
        default=None,
        choices=["full_shard", "shard_grad_op", "no_shard", "hybrid_shard"],
        help="FSDP sharding strategy (overrides config)",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        default=False,
        help="Enable debug mode: reduces buffer size, sets log/save path to debug.",
    )
    parser.add_argument(
        "--wandb_offline",
        type=str,
        default=None,
        help="Whether to run wandb in offline mode (overrides config).",
    )
    parser.add_argument(
        "--visualize",
        action="store_true",
        default=False,
        help="Whether to visualize samples during training.",
    )
    parser.add_argument(
        "--log_to_file",
        action="store_true",
        default=False,
        help="Save entry logs, warnings and errors alongside native logs/rank_N.log.",
    )
    return parser.parse_args()


def load_config(config_path: str, cli_args=None):
    """Load configuration from YAML file into TrainConfig."""
    return load_typed_config(config_path, cli_args=cli_args)


def setup_logger(cfg):
    """Setup wandb logger if enabled"""
    log = cfg.logging
    if log.use_wandb and is_main_process():
        logger.info(
            "rank %s is initializing wandb , is main process %s",
            torch.distributed.get_rank(),
            is_main_process(),
        )
        wandb_run = wandb.init(
            project=log.log_project,
            name=log.log_name,
            entity=log.log_entity,
            config=dataclasses.asdict(cfg),
            save_code=False,
            force=False,
            mode="offline" if log.wandb_offline else "online",
        )
        logger.info("Wandb Initialized")
        return wandb_run
    return None


def print_fsdp_config(cfg):
    """Print FSDP configuration"""
    if is_main_process():
        dist = cfg.distributed
        logger.info("%s", "=" * 60)
        logger.info("FSDP Configuration:")
        logger.info("  use_fsdp: %s", dist.use_fsdp)
        logger.info("%s", "=" * 60)


# Install the video reader before importing the RoboTwin trainer/data modules.
install_bounded_video_reader()
from wall_x.robotwin.trainer import RobotwinFSDPTrainer  # noqa: E402


def atomic_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(path)


class RoboTwinTrainer(RobotwinFSDPTrainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.robotwin_started = time.monotonic()
        self.robotwin_run_dir = (
            Path(os.environ["ROBOTWIN_RUN_DIR"])
            if os.environ.get("ROBOTWIN_RUN_DIR")
            else None
        )
        if self.robotwin_run_dir is not None and self.rank == 0:
            dataset_name = self.cfg.data.lerobot_config["repo_id"]
            assert list(self.normalizer_action.min) == [dataset_name]
            assert list(self.normalizer_propri.min) == [dataset_name]
            normalizers = {}
            for name, normalizer in [
                ("action", self.normalizer_action),
                ("state", self.normalizer_propri),
            ]:
                normalizers[name] = {
                    "min": normalizer.min[dataset_name].detach().cpu().tolist(),
                    "delta": normalizer.delta[dataset_name].detach().cpu().tolist(),
                }
            atomic_json(
                self.robotwin_run_dir / "initialized-normalizers.json", normalizers
            )
            self._write_progress("running", self.global_step)

    def _write_progress(self, state, updates):
        import torch

        atomic_json(
            self.robotwin_run_dir / "training-progress.json",
            {
                "state": state,
                "completed_updates": updates,
                "target_updates": self.num_training_steps,
                "time": time.time(),
                "elapsed_training_seconds": time.monotonic() - self.robotwin_started,
                "learning_rate": self.get_lr(),
                "max_allocated_gib": torch.cuda.max_memory_allocated() / 2**30,
                "max_reserved_gib": torch.cuda.max_memory_reserved() / 2**30,
            },
        )

    def optimizer_step(self):
        super().optimizer_step()
        updates = self.global_step + 1
        if (
            self.robotwin_run_dir is not None
            and self.rank == 0
            and (updates % 25 == 0 or updates == 1)
        ):
            self._write_progress("running", updates)

    def save_checkpoint(self, epoch, step=0):
        super().save_checkpoint(epoch, step)
        if self.robotwin_run_dir is None or self.rank != 0:
            return
        checkpoint = Path(self.cfg.checkpoint.save_path) / (
            f"{epoch}_{step}" if step else str(epoch)
        )
        model = checkpoint / "model.safetensors"
        from safetensors import safe_open

        with safe_open(str(model), framework="pt", device="cpu") as weights:
            tensor_count = len(list(weights.keys()))
            assert tensor_count > 100
        updates = self.global_step + int(self._inside_train_loop)
        atomic_json(
            self.robotwin_run_dir / "checkpoint-ready" / f"{updates:08d}.json",
            {
                "completed_updates": updates,
                "checkpoint": str(checkpoint),
                "model_size": model.stat().st_size,
                "tensor_count": tensor_count,
                "saved_at": time.time(),
                "norm_stats_sha256": hashlib.sha256(
                    Path(self.cfg.data.norm_stats_path).read_bytes()
                ).hexdigest(),
                "target_updates": self.num_training_steps,
            },
        )


def main():
    args = parse_args()

    with training_logging(args.config, enabled=args.log_to_file, debug=args.debug):
        wandb_run = None
        try:
            torch.cuda.init()
            local_rank = int(os.environ.get("LOCAL_RANK", 0))
            device = torch.device(f"cuda:{local_rank}")
            torch.cuda.set_device(device)
            torch.distributed.init_process_group("nccl", device_id=device)
            cfg = load_config(args.config, cli_args=args)
            wandb_run = setup_logger(cfg)
            print_fsdp_config(cfg)
            trainer = RoboTwinTrainer(
                train_config=cfg,
                wandb_run=wandb_run,
            )
            trainer.fit()
        finally:
            cleanup_distributed()
            if wandb_run is not None:
                wandb_run.finish()


if __name__ == "__main__":
    main()
