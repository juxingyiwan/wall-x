"""B601 adaptation around the unchanged native LeRobot dataset interfaces.

The native dataset, collator, image/text preprocessing and sqrt sampler remain
reusable. This module owns episode selection, command encoding and unclipped
normalization, and returns the same dataset/dataloader contracts as the backend.
"""

import copy
import logging
import os

import numpy as np
import torch
from lerobot.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata
from torch.utils.data import DataLoader, DistributedSampler, Subset

from wall_x.data.b601 import (
    absolute_window,
    codec_from_config,
    load_b601_norm_stats,
    settings,
    validate_statistics,
)
from wall_x.data.b601_split import resolve_split
from wall_x.data.backends.lerobot.config import LerobotConfig
from wall_x.data.backends.lerobot.loader import (
    DataCollator,
    PreprocessedDataset,
    get_data_configs,
)
from wall_x.model.core.action.affine import scale

logger = logging.getLogger(__name__)


class _CameraOrderMetadata:
    def __init__(self, metadata, camera_keys):
        self._metadata = metadata
        self.camera_keys = list(camera_keys)

    def __getattr__(self, name):
        return getattr(object.__getattribute__(self, "_metadata"), name)


class _CameraOrderDataset:
    """Read-through metadata view; never mutate the source dataset."""

    def __init__(self, dataset, camera_keys):
        self._source = dataset
        self.meta = _CameraOrderMetadata(dataset.meta, camera_keys)

    def __getattr__(self, name):
        return getattr(object.__getattribute__(self, "_source"), name)

    def __len__(self):
        return len(self._source)

    def __getitem__(self, index):
        return self._source[index]


class B601PreprocessedDataset(PreprocessedDataset):
    def __init__(
        self,
        dataset,
        config,
        norm_stats,
        dataload_config,
        lerobot_config,
        seed=42,
        rank=0,
        world_size=1,
    ):
        opts = settings(config)
        if opts is None:
            raise ValueError("B601 adapter requires explicit B601 settings")
        view = _CameraOrderDataset(dataset, dataload_config["key_mappings"]["camera"])
        # The existing test_only switch skips the native implicit random split.
        super().__init__(
            view,
            config,
            norm_stats,
            dataload_config,
            lerobot_config,
            seed=seed,
            rank=rank,
            world_size=world_size,
            test_only=True,
        )
        self.b601_codec = codec_from_config(config)
        self.data_config = LerobotConfig.from_yaml_dict(config)
        split = resolve_split(config)
        numeric = dataset.hf_dataset
        if hasattr(numeric, "with_format"):
            numeric = numeric.select_columns(
                [
                    "episode_index",
                    "frame_index",
                    "task_index",
                    dataload_config["key_mappings"]["action"],
                ]
            ).with_format("numpy")[:]
        self.b601_episode_ids = np.asarray(
            numeric["episode_index"], dtype=np.int64
        ).reshape(-1)
        self._frame_ids = np.asarray(numeric["frame_index"], dtype=np.int64).reshape(-1)
        actions = np.asarray(
            numeric[dataload_config["key_mappings"]["action"]], dtype=np.float32
        )
        self.b601_actions = {}
        for episode in np.unique(self.b601_episode_ids):
            rows = np.flatnonzero(self.b601_episode_ids == episode)
            if not np.array_equal(self._frame_ids[rows], np.arange(len(rows))):
                raise ValueError("B601 episodes must have contiguous ordered frames")
            self.b601_actions[int(episode)] = actions[rows]
        train_rows = np.flatnonzero(
            np.isin(self.b601_episode_ids, split["train_episodes"])
        ).tolist()
        val_rows = np.flatnonzero(
            np.isin(self.b601_episode_ids, split["validation_episodes"])
        ).tolist()
        if not train_rows or len(train_rows) + len(val_rows) != len(dataset):
            raise ValueError("B601 episode manifest does not cover dataset")
        self.b601_train_task_ids = (
            np.asarray(numeric["task_index"])[train_rows].reshape(-1).tolist()
        )
        self.train_dataset = Subset(dataset, train_rows)
        self.val_dataset = Subset(dataset, val_rows)
        self._train()

    def __getitem__(self, index):
        # Native preprocessing reads each video frame only once. Replace its
        # label with the shared full-window encoding from cached numeric columns.
        result = super().__getitem__(index)
        row = self._dataset.indices[index]
        raw = absolute_window(
            self.b601_actions[int(self.b601_episode_ids[row])],
            int(self._frame_ids[row]),
            self.dataload_config["action_horizon"] - 1,
        )
        anchor = self.b601_codec.anchor_from_state(result["agent_pos"])
        result["action"] = torch.from_numpy(
            self.b601_codec.encode_absolute_commands(raw, anchor)
        )
        return result

    def _make_dataloader(self, training):
        self._train() if training else self._eval()
        task_sampling = self.dataload_config.get("task_sampling") if training else None
        if task_sampling == "sqrt":
            from wall_x.data.sampling import DistributedSquareRootSampler

            sampler = DistributedSquareRootSampler(
                self.b601_train_task_ids,
                num_replicas=self.world_size,
                rank=self.rank,
                seed=self.seed,
            )
            if self.rank == 0:
                logger.info(
                    "Square-root task sampling: counts=%s probabilities=%s",
                    dict(sampler.counts),
                    sampler.probabilities,
                )
        elif task_sampling is None:
            sampler = DistributedSampler(
                self,
                num_replicas=self.world_size,
                rank=self.rank,
                shuffle=training,
                seed=self.seed,
                drop_last=training,
            )
        else:
            raise ValueError(f"Unsupported data.task_sampling: {task_sampling!r}")
        workers = self.config.get("num_workers", 4)
        batch_size = self.config.get("batch_size_per_gpu", 8)
        if not training:
            batch_size = self.config.get("eval_batch_size_per_gpu", batch_size)
        loader = DataLoader(
            self,
            batch_size=batch_size,
            sampler=sampler,
            num_workers=workers,
            collate_fn=B601DataCollator(
                self.config, self.dataload_config, self.norm_stats, self.lerobot_config
            ),
            pin_memory=True,
            persistent_workers=workers > 0,
            prefetch_factor=2 if workers > 0 else None,
            drop_last=training,
        )
        return loader, sampler

    def get_train_dataloader(self):
        return self._make_dataloader(True)

    def get_val_dataloader(self):
        if not len(self.val_dataset):
            return None, None
        return self._make_dataloader(False)


class B601DataCollator(DataCollator):
    def __init__(self, config, dataload_config, stats, lerobot_config):
        self.b601_options = settings(config)
        if self.b601_options is None:
            raise ValueError("B601 collator requires explicit B601 settings")
        super().__init__(config, dataload_config, stats, lerobot_config)
        self._b601_rng_context = None

    def state_dict(self):
        return {
            "context": self._b601_rng_context,
            "generator": copy.deepcopy(self.np_rng.bit_generator.state),
            "sampling": self._sampling_settings(),
        }

    def _sampling_settings(self):
        return {
            key: float(getattr(self, key))
            for key in ("beta_alpha", "beta_beta", "time_shift", "s")
        }

    def load_state_dict(self, state):
        if state.get("sampling") != self._sampling_settings():
            raise ValueError("B601 collator time-sampling settings mismatch")
        context = state["context"]
        if context is not None:
            rank = (
                torch.distributed.get_rank()
                if torch.distributed.is_initialized()
                else 0
            )
            expected = (int(self.dataload_config.get("seed", 42)), rank, 0)
            if tuple(context) != expected:
                raise ValueError("B601 collator RNG seed/rank/worker mismatch")
        self.np_rng.bit_generator.state = copy.deepcopy(state["generator"])
        self._b601_rng_context = tuple(context) if context is not None else None

    def _normalize_field(self, value, lower, span, field):
        if value.shape[-1] != 7 or not torch.isfinite(value).all():
            raise ValueError("B601 inputs require seven finite real dimensions")
        return scale(value, lower, span, clip=self.b601_options[field])

    @classmethod
    def _normalize(cls, value, lower, span):
        # __call__ has already normalized both fields with their own clip policy.
        # Native collation still owns masks, padding, image/text and tensor layout.
        return value

    def __call__(self, batch):
        normalized = []
        for item in batch:
            converted = dict(item)
            converted["agent_pos"] = self._normalize_field(
                item["agent_pos"],
                self.state_min_stat,
                self.state_delta,
                "clip_state_inputs",
            )
            converted["action"] = self._normalize_field(
                item["action"],
                self.action_min_stat,
                self.action_delta,
                "clip_action_targets",
            )
            normalized.append(converted)
        return super().__call__(normalized)

    def sample_time(self, batch_size, device, dtype):
        worker = torch.utils.data.get_worker_info()
        rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else 0
        context = (
            int(self.dataload_config.get("seed", 42)),
            rank,
            worker.id if worker else 0,
        )
        if self._b601_rng_context != context:
            self.np_rng = np.random.default_rng(np.random.SeedSequence(context))
            self._b601_rng_context = context
        return super().sample_time(batch_size, device, dtype)


def load_b601_data(config, lerobot_config, rank=0, world_size=1, seed=42):
    """Return the native backend contract, applying B601 selection only here."""
    torch.manual_seed(seed)
    opts = settings(config)
    if opts is None:
        raise ValueError("B601 adapter requires explicit B601 settings")
    if lerobot_config.get("episodes") is not None:
        raise ValueError("B601 episode selection is controlled by its split settings")
    stats_path = config["norm_stats_path"]
    split = resolve_split(config)
    validate_statistics(config, stats_path, split_path=split)
    dataload_config = get_data_configs(config["data"])
    stats = load_b601_norm_stats(stats_path, dataload_config["key_mappings"])
    repo_id, root = lerobot_config["repo_id"], lerobot_config.get("root")
    metadata = LeRobotDatasetMetadata(repo_id, root=root)
    if metadata.fps != opts["fps"]:
        raise ValueError("B601 dataset/checkpoint frequency mismatch")
    selected = sorted(split["train_episodes"] + split["validation_episodes"])
    distributed = torch.distributed.is_initialized()
    local_rank = int(os.environ.get("LOCAL_RANK", 0)) if distributed else 0
    local_world_size = int(os.environ.get("LOCAL_WORLD_SIZE", 1)) if distributed else 1
    for rank_on_node in range(local_world_size):
        if local_rank == rank_on_node:
            raw = LeRobotDataset(
                repo_id=repo_id,
                root=root,
                episodes=selected,
                delta_timestamps=None,
                video_backend="pyav",
            )
        if distributed:
            torch.distributed.barrier(device_ids=[local_rank])
    dataset = B601PreprocessedDataset(
        raw,
        config,
        stats,
        dataload_config,
        lerobot_config,
        seed=seed,
        rank=rank,
        world_size=world_size,
    )
    if rank == 0:
        logger.info(
            "B601 episode split: train=%d (%d frames), validation=%d (%d frames)",
            len(split["train_episodes"]),
            len(dataset.train_dataset),
            len(split["validation_episodes"]),
            len(dataset.val_dataset),
        )
    return dataset, len(dataset) // world_size // config.get("batch_size_per_gpu", 8)
