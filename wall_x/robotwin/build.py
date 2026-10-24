"""RoboTwin trainer/data bridge over the unchanged LeRobot configuration."""

import logging

import torch.distributed as dist

from wall_x.data.backends.lerobot.bridge import (
    build_dataset_config as _native_flat_config,
)
from wall_x.data.backends.lerobot.bridge import wrap_dataset
from wall_x.robotwin.dataset import load_robotwin_data

logger = logging.getLogger(__name__)


def _build_flat_config(cfg):
    flat, lerobot = _native_flat_config(cfg)
    if lerobot.get("adapter") != "robotwin":
        raise ValueError("RoboTwin entry requires data.lerobot_config.adapter=robotwin")
    raw = getattr(cfg, "_raw_data", {}) or {}
    flat["data"]["episode_split_path"] = getattr(
        cfg.data, "episode_split_path", None
    ) or raw.get("episode_split_path")
    if "clip_action_targets" in raw:
        flat["data"]["clip_action_targets"] = raw["clip_action_targets"]
    return flat, lerobot


def load_robotwin_v2(
    cfg,
):
    """Build lerobot (wrapper, dataloader, train_num) from TrainConfig.

    The third return value ``train_num`` is a snapshot of
    ``len(train_dataloader)`` at construction time. It matches
    ``wrapper.global_train_iters.value`` initially but does not track
    subsequent rebuilds inside ``set_epoch`` - callers doing dynamic
    resampling should read from the mp.Value, not from this snapshot.
    """
    flat_cfg, lerobot_cfg = _build_flat_config(cfg)

    if dist.is_initialized():
        rank = dist.get_rank()
        world_size = dist.get_world_size()
    else:
        rank = 0
        world_size = 1

    seed = cfg.hyperparams.seed
    inner, _ = load_robotwin_data(
        flat_cfg,
        lerobot_cfg,
        rank=rank,
        world_size=world_size,
        seed=seed,
    )

    # PreprocessedDataset.get_*_dataloader returns (dataloader, sampler).
    # Build val first, train second, so the inner ``_dataset`` pointer is
    # left at the train split when we finish - workers fork from that
    # state on first iteration.
    val_dataloader, _ = inner.get_val_dataloader()
    val_num = len(val_dataloader) if val_dataloader is not None else 0

    train_dataloader, train_sampler = inner.get_train_dataloader()
    train_num = len(train_dataloader)

    if rank == 0:
        logger.info(
            "\n%s\nLeRobot Data Loading Configuration:\n"
            "  RANK: %d\n  WORLD SIZE: %d\n"
            "  BATCH SIZE PER DEVICE: %d\n  GLOBAL BATCH SIZE: %d\n"
            "  TRAIN BATCHES: %d\n  VAL BATCHES: %d\n"
            "  NUM WORKERS: %d\n  REPO ID: %s\n%s",
            "=" * 50,
            rank,
            world_size,
            flat_cfg["batch_size_per_gpu"],
            flat_cfg["batch_size_per_gpu"] * world_size,
            train_num,
            val_num,
            flat_cfg["num_workers"],
            lerobot_cfg.get("repo_id"),
            "=" * 50,
        )

    wrapper = wrap_dataset(
        inner,
        train_dataloader,
        train_sampler,
        train_num,
        val_dataloader=val_dataloader,
        val_num=val_num,
    )
    return wrapper, train_dataloader, train_num
