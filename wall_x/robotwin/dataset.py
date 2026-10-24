"""RoboTwin initialization using LeRobot's native cache and Wall-X preprocessing."""

import json
import logging
import time
from copy import copy
from pathlib import Path

import torch
from lerobot.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata
from torch.utils.data import Subset

from wall_x.data.backends.lerobot.loader import PreprocessedDataset, get_data_configs
from wall_x.data.backends.lerobot.utils import load_norm_stats
from wall_x.robotwin.episode_split import (
    load_episode_split,
    split_frame_indices,
)
from wall_x.robotwin.video import install_bounded_video_reader

logger = logging.getLogger(__name__)


class RobotwinPreprocessedDataset(PreprocessedDataset):
    """Keep square-image behavior local to the RoboTwin adapter."""

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
        test_only=False,
    ):
        # Bypass the shared loader's frame-wise random split.
        super().__init__(
            dataset,
            config,
            norm_stats,
            dataload_config,
            lerobot_config,
            seed=seed,
            rank=rank,
            world_size=world_size,
            test_only=True,
        )
        if not test_only:
            if not dataload_config.get("episode_split_path"):
                raise ValueError("RoboTwin requires an explicit episode_split_path")
            train, val = split_frame_indices(
                dataset.hf_dataset["episode_index"],
                dataload_config["episode_split_path"],
            )
            self.train_dataset, self.val_dataset = Subset(dataset, train), Subset(
                dataset, val
            )
            self._train()

    def _isolated_dataloader(self, training):
        from wall_x.robotwin.collator import RobotwinDataCollator

        # The unmodified native implementation switches self._dataset. Give
        # each loader its own shallow copy before it creates the sampler.
        owned = copy(self)
        parent = super(RobotwinPreprocessedDataset, owned)
        loader, sampler = (
            parent.get_train_dataloader() if training else parent.get_val_dataloader()
        )
        loader.collate_fn = RobotwinDataCollator(
            self.config, self.dataload_config, self.norm_stats, self.lerobot_config
        )
        from wall_x.robotwin.resume import ReplayableLoader

        loader = ReplayableLoader(
            loader, self.seed, self.rank, "train" if training else "validation"
        )
        return loader, sampler

    def get_train_dataloader(self):
        return self._isolated_dataloader(True)

    def get_val_dataloader(self):
        return self._isolated_dataloader(False)

    def _vision_preprocess(self, frames):
        if not self.lerobot_config.get("pad_to_square", False):
            return super()._vision_preprocess(frames)

        from PIL import Image

        from wall_x.robotwin.images import resize_robotwin_square

        images = []
        for key in self.hf_dataset.meta.camera_keys:
            pixels = frames[key].clone().permute(1, 2, 0)
            image = Image.fromarray((pixels * 255).to(torch.uint8).cpu().numpy())
            orig_width, orig_height = image.size
            image = resize_robotwin_square(
                image,
                self.data_config.resolution[self._cam_key_mapping[key]],
                factor=self.data_config.image_factor,
                min_pixels=self.data_config.min_pixels,
                max_pixels=self.data_config.max_pixels,
            )
            images.append(image)
        return images, orig_height, orig_width, image.height, image.width


def load_robotwin_data(config, lerobot_config, rank=0, world_size=1, seed=42):
    """Initialize each rank independently, preserving the explicit episode split.

    LeRobot uses the Hugging Face Datasets file lock when preparing the Arrow
    cache. Subsequent readers memory-map it independently. Video presence checks
    remain enabled; no process mutates source videos or skips missing data.
    """
    started = time.perf_counter()
    torch.manual_seed(seed)
    # Eight ranks with multiple workers must not each spawn automatic AV1/RGB pools.
    install_bounded_video_reader()
    data_config = get_data_configs(config["data"])
    split_path = data_config.get("episode_split_path")
    if not split_path:
        raise ValueError("RoboTwin loading requires data.episode_split_path")
    repo_id = lerobot_config["repo_id"]
    root = lerobot_config.get("root")
    meta = LeRobotDatasetMetadata(repo_id, root=root)
    available = lerobot_config.get("episodes")
    if available is None:
        available = list(range(meta.total_episodes))
    train_episodes, validation_episodes = load_episode_split(split_path, available)
    if lerobot_config.get("expected_tasks"):
        from wall_x.robotwin.task_manifest import validate_task_selection

        provenance = json.loads(
            (Path(root) / "meta/robotwin_provenance.json").read_text()
        )["episodes"]
        validate_task_selection(
            provenance, train_episodes, lerobot_config["expected_tasks"]
        )
    selected = sorted(train_episodes + validation_episodes)
    # Passing a full list makes LeRobot build a redundant dictionary for every
    # frame. Its native full-dataset mode already uses the absolute row indices.
    episodes = None if selected == list(range(meta.total_episodes)) else selected
    key_mappings = data_config["key_mappings"]
    task = config.get("task") or {}
    norm_stats = load_norm_stats(
        config["norm_stats_path"],
        key_mappings,
        dof_config=config.get("dof_config") or task.get("dof_config", {}),
        agent_pos_config=config.get("agent_pos_config")
        or task.get("agent_pos_config", {}),
    )
    logger.info("[RoboTwin rank %s] Initializing LeRobot data concurrently", rank)
    source = LeRobotDataset(
        repo_id=repo_id,
        root=root,
        episodes=episodes,
        delta_timestamps={
            key_mappings["action"]: [
                t / meta.fps for t in range(data_config["action_horizon"] - 1)
            ]
        },
        video_backend="pyav",
    )
    # A formatted copy reads only the numeric column, in one batch, while the
    # source keeps its original torch transform for training samples.
    episode_ids = source.hf_dataset.with_format("numpy", columns=["episode_index"])[:][
        "episode_index"
    ]
    train_indices, validation_indices = split_frame_indices(episode_ids, split_path)
    # Initialize the shared preprocessing without letting it split the dataset;
    # attach the validated whole-episode subsets explicitly below.
    dataset = RobotwinPreprocessedDataset(
        source,
        config,
        norm_stats,
        data_config,
        lerobot_config,
        seed=seed,
        rank=rank,
        world_size=world_size,
        test_only=True,
    )
    dataset.train_dataset = Subset(source, train_indices)
    dataset.val_dataset = Subset(source, validation_indices)
    dataset._train()
    train_num = len(train_indices) // world_size // config.get("batch_size_per_gpu", 8)
    logger.info(
        "[RoboTwin rank %s] Data ready in %.2fs: train=%s episodes/%s frames, "
        "validation=%s episodes/%s frames",
        rank,
        time.perf_counter() - started,
        len(train_episodes),
        len(train_indices),
        len(validation_episodes),
        len(validation_indices),
    )
    return dataset, train_num
