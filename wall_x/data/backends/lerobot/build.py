"""LeRobot data loading bridge for typed training configs."""

from __future__ import annotations

import logging
import multiprocessing as mp
from typing import Any, Dict, Tuple

import torch
import torch.distributed as dist

from wall_x.data.backends.lerobot.config import LerobotConfig
from wall_x.data.backends.lerobot.utils import load_norm_stats
from wall_x.model.core.action.normalizer import (
    create_normalizers_from_lerobot_norm_stats,
)

logger = logging.getLogger(__name__)


def load_lerobot_normalizers(cfg: Any):
    """Create model normalizers from the LeRobot norm stats configured for a run."""
    data = cfg.data

    norm_stats_path = getattr(data, "norm_stats_path", None)
    if not norm_stats_path:
        return None

    key_mappings = getattr(data, "key_mappings", None)
    if not key_mappings:
        raise ValueError(
            "LeRobot normalizer from norm_stats_path requires data.key_mappings"
        )

    lerobot_config = getattr(data, "lerobot_config", None)
    if not isinstance(lerobot_config, dict) or not lerobot_config.get("repo_id"):
        raise ValueError(
            "LeRobot normalizer from norm_stats_path requires "
            "data.lerobot_config.repo_id"
        )

    from wall_x.data.b601 import (
        configure_normalizers,
        load_b601_norm_stats,
        settings,
        validate_statistics,
    )

    raw_config = dict(getattr(cfg, "_raw_yaml", {}) or {})
    opts = settings(raw_config)
    validate_statistics(
        raw_config, norm_stats_path, split_path=opts["split_manifest"] if opts else None
    )
    dataset_name = str(lerobot_config["repo_id"])
    if "." in dataset_name:
        raise ValueError(
            "repo_id is the normalizer key and cannot contain '.'. Use a stable ID (e.g. libero_all) and put the dataset path in lerobot_config.root. Existing checkpoint keys must not be renamed when resuming."
        )
    if opts is not None:
        norm_stats = load_b601_norm_stats(norm_stats_path, key_mappings)
    else:
        norm_stats = load_norm_stats(
            norm_stats_path,
            key_mappings,
            dof_config=dict(cfg.task.dof_config or {}),
            agent_pos_config=dict(cfg.task.agent_pos_config or {}),
        )
    normalizer_action, normalizer_propri = create_normalizers_from_lerobot_norm_stats(
        norm_stats,
        dataset_name,
        cfg.action_dim,
        cfg.propri_dim,
    )
    configure_normalizers(normalizer_action, normalizer_propri, raw_config)
    return normalizer_action, normalizer_propri, norm_stats_path, dataset_name


class _LerobotDatasetWrapper:
    """Trainer-facing wrapper aligning PreprocessedDataset with v1 API.

    Train and validation DataLoaders own independent split views.

    This wrapper:
    - Caches the rebuilt train dataloader / sampler so
      ``set_epoch(epoch)`` can reset shuffling per-epoch.
    - Owns the val dataloader so the trainer's ``val_loop`` can do
      ``self.dataset.get_val_dataloader()`` and iterate directly (matching
      what the v1/v2 wrappers return).
    """

    def __init__(
        self,
        inner,
        train_dataloader: torch.utils.data.DataLoader,
        train_sampler,
        train_num: int,
        val_dataloader: torch.utils.data.DataLoader = None,
        val_num: int = 0,
    ):
        self._inner = inner
        self._train_dataloader = train_dataloader
        self._train_sampler = train_sampler
        self._train_num = train_num
        self._val_dataloader = val_dataloader
        self.global_train_iters = mp.Value("i", train_num)
        self.global_val_iters = mp.Value("i", val_num)

    def __len__(self) -> int:
        return self._train_num

    def get_train_dataloader(self):
        return self._train_dataloader

    def get_val_dataloader(self):
        return self._val_dataloader

    def set_epoch(self, epoch: int) -> None:
        """Seed the per-epoch shuffle in the train DistributedSampler."""
        if self._train_sampler is not None and hasattr(
            self._train_sampler, "set_epoch"
        ):
            self._train_sampler.set_epoch(epoch)


def load_trainer_data_config(cfg: Any) -> LerobotConfig:
    """Build the inference/trainer data config from a typed TrainConfig."""
    raw_yaml = dict(getattr(cfg, "_raw_yaml", {}) or {})
    raw_data = dict(getattr(cfg, "_raw_data", {}) or {})
    data = getattr(cfg, "data", None)

    data_section = dict(raw_yaml.get("data", {}) or {})
    data_section.update(raw_data)

    if data is not None:
        for key in (
            "resolution",
            "train_test_split",
            "priority_order",
            "camera_name_mapping",
        ):
            value = getattr(data, key, None)
            if value is not None:
                data_section.setdefault(key, value)

    raw_yaml["data"] = data_section
    raw_yaml.setdefault("model_type", getattr(cfg, "model_type", "qwen2_5"))
    return load_trainer_data_config_from_yaml_dict(raw_yaml)


def load_trainer_data_config_from_yaml_dict(yaml_dict: Dict[str, Any]) -> LerobotConfig:
    """Build the LeRobot runtime config from a raw training YAML dict."""
    return LerobotConfig.from_yaml_dict(yaml_dict)


def _build_flat_config(cfg: Any) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Map typed TrainConfig -> (flat_config, lerobot_config) for legacy entry.

    ``load_lerobot_data`` expects a 2509-style flat dict plus a separate
    ``lerobot_config`` carrying ``repo_id`` / ``root``. This function is
    the one place that translation lives; keep it surgical so future
    field additions on ``LeRobotDataConfig`` do not require touching the
    legacy loader.
    """
    model = cfg.model
    data = cfg.data
    hp = cfg.hyperparams
    raw = getattr(cfg, "_raw_yaml", {}) or {}
    raw_data = dict(getattr(cfg, "_raw_data", {}) or {})

    lerobot_cfg = dict(data.lerobot_config or {})
    if "repo_id" not in lerobot_cfg:
        raise ValueError(
            "lerobot requires data.lerobot_config.repo_id to be set "
            "(HuggingFace LeRobot dataset id)."
        )

    data_section: Dict[str, Any] = {
        "key_mappings": data.key_mappings,
        "action_horizon": cfg.task.action_horizon,
        "train_test_split": data.train_test_split,
        "seed": hp.seed,
        "resolution": data.resolution,
    }
    if raw_data.get("max_length") is not None:
        data_section["max_length"] = raw_data["max_length"]
    for key in ("min_pixels", "max_pixels", "image_factor"):
        if key in raw_data:
            data_section[key] = raw_data[key]
    if data.task_sampling is not None:
        data_section["task_sampling"] = data.task_sampling
    if data.priority_order is not None:
        data_section["priority_order"] = data.priority_order
    if data.camera_name_mapping is not None:
        data_section["camera_name_mapping"] = data.camera_name_mapping
    data_section.setdefault(
        "use_state_string_representation",
        cfg.task.use_state_string_representation,
    )
    data_section.setdefault(
        "state_bins",
        raw_data.get("state_bins", raw.get("state_bins", 256)),
    )

    # Dof/agent_pos totals for the collator's zero-pad step. When resuming
    # from a checkpoint trained on a larger action space, task.dof_config
    # should include an ``action_padding`` key that absorbs the diff; the
    # collator right-pads action/agent_pos tensors to these totals with
    # dof_mask/agent_pos_mask zeroed on padded dims so loss doesn't flow
    # through them.
    dof_total = int(sum((cfg.task.dof_config or {}).values()))
    agent_pos_total = int(sum((cfg.task.agent_pos_config or {}).values()))

    flat: Dict[str, Any] = {
        "model_type": cfg.model_type,
        "processor_path": getattr(model, "processor_path", "") or "",
        "norm_stats_path": data.norm_stats_path or raw.get("norm_stats_path"),
        "batch_size_per_gpu": hp.batch_size_per_gpu,
        "eval_batch_size_per_gpu": raw.get(
            "eval_batch_size_per_gpu", hp.batch_size_per_gpu
        ),
        "num_workers": data.num_workers,
        "padding_side": data.padding_side,
        "use_fast_tokenizer": data.use_fast_tokenizer,
        "action_tokenizer_path": data.action_tokenizer_path,
        "noise_scheduler": data.noise_scheduler or {},
        "dof_total_dim": dof_total,
        "agent_pos_total_dim": agent_pos_total,
        "dof_config": dict(cfg.task.dof_config or {}),
        "agent_pos_config": dict(cfg.task.agent_pos_config or {}),
        "use_state_string_representation": cfg.task.use_state_string_representation,
        "state_bins": int(
            raw_data.get("state_bins")
            or data_section.get("state_bins")
            or raw.get("state_bins")
            or 256
        ),
        "data": {**data_section, "lerobot_config": lerobot_cfg},
    }
    return flat, lerobot_cfg


def load_lerobot_v2(
    cfg: Any,
) -> Tuple[_LerobotDatasetWrapper, torch.utils.data.DataLoader, int]:
    """Build lerobot (wrapper, dataloader, train_num) from TrainConfig.

    The third return value ``train_num`` is a snapshot of
    ``len(train_dataloader)`` at construction time. It matches
    ``wrapper.global_train_iters.value`` initially but does not track
    subsequent rebuilds inside ``set_epoch`` - callers doing dynamic
    resampling should read from the mp.Value, not from this snapshot.
    """
    from wall_x.data.backends.lerobot.episode_split import validate_resume_split
    from wall_x.data.backends.lerobot.loader import load_lerobot_data

    flat_cfg, lerobot_cfg = _build_flat_config(cfg)
    if lerobot_cfg.get("b601") is None:
        validate_resume_split(
            cfg.checkpoint.resume_from, lerobot_cfg, cfg.data.train_test_split
        )

    if dist.is_initialized():
        rank = dist.get_rank()
        world_size = dist.get_world_size()
    else:
        rank = 0
        world_size = 1

    seed = cfg.hyperparams.seed
    loader = load_lerobot_data
    if lerobot_cfg.get("b601") is not None:
        from wall_x.data.backends.lerobot.b601_adapter import load_b601_data

        loader = load_b601_data
    inner, _ = loader(
        flat_cfg,
        lerobot_cfg,
        rank=rank,
        world_size=world_size,
        seed=seed,
    )

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

    wrapper = _LerobotDatasetWrapper(
        inner,
        train_dataloader,
        train_sampler,
        train_num,
        val_dataloader=val_dataloader,
        val_num=val_num,
    )
    return wrapper, train_dataloader, train_num


def build(cfg, ctx):
    """Backend Protocol entry - returns a ``DataBundle``.

    Wraps ``load_lerobot_v2`` (which returns the trainer-facing triple)
    into the unified ``DataBundle`` shape every backend exposes.
    """
    from wall_x.data._bundle import DataBundle

    wrapper, train_dataloader, train_num = load_lerobot_v2(cfg)

    val_loader = wrapper.get_val_dataloader()

    return DataBundle(
        dataset=wrapper,
        train_loader=train_dataloader,
        val_loader=val_loader,
        train_iters=train_num,
        val_iters=wrapper.global_val_iters.value,
        set_epoch=wrapper.set_epoch,
    )
