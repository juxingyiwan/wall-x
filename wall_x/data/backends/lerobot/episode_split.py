"""One episode split contract for training, validation and normalization."""

import json
import math
from pathlib import Path


def episode_split(total_episodes, lerobot_config, train_ratio):
    """Keep configured episode order; reserve its tail for validation."""
    if not math.isfinite(train_ratio) or not 0 < train_ratio <= 1:
        raise ValueError("train_test_split must be in (0, 1]")
    episodes = lerobot_config.get("episodes")
    episodes = list(range(total_episodes)) if episodes is None else list(episodes)
    if (
        not episodes
        or any(
            not isinstance(ep, int)
            or isinstance(ep, bool)
            or not 0 <= ep < total_episodes
            for ep in episodes
        )
        or len(set(episodes)) != len(episodes)
    ):
        raise ValueError(
            "episodes must be nonempty, unique integer IDs within the dataset"
        )
    count = int(len(episodes) * train_ratio)
    if count == 0:
        raise ValueError(
            "No training episodes; use more episodes or train_test_split=1"
        )
    return {
        "version": 1,
        "repo_id": str(lerobot_config["repo_id"]),
        "train_episodes": episodes[:count],
        "val_episodes": episodes[count:],
    }


def split_mode(lerobot_config):
    mode = lerobot_config.get("split_mode", "episode")
    if mode not in {"episode", "legacy_frame"}:
        raise ValueError("lerobot_config.split_mode must be episode or legacy_frame")
    return mode


def validate_stats_split(path, manifest):
    recorded = json.loads(Path(path).read_text(encoding="utf-8")).get("episode_split")
    if recorded != manifest:
        raise ValueError(
            "Normalization statistics do not match the training episode split. "
            "Recompute with compute_norm_stats.py --train_config CONFIG. "
            "Use split_mode: legacy_frame only to reproduce an old training run."
        )


def validate_resume_split(resume_path, lerobot_config, train_ratio):
    """Do not silently reinterpret the split of an existing training checkpoint."""
    if not resume_path or not Path(resume_path).is_dir():
        return
    import yaml

    config_path = Path(resume_path) / "config.yml"
    if not config_path.is_file():
        raise ValueError(
            "A training resume requires checkpoint config.yml for split validation"
        )
    old = (
        yaml.load(config_path.read_text(encoding="utf-8"), Loader=yaml.FullLoader) or {}
    )
    old_data = old.get("data", {})
    old_lerobot = old_data.get("lerobot_config", {})
    old_mode = old_lerobot.get("split_mode", "legacy_frame")
    if old_mode != split_mode(lerobot_config):
        raise ValueError(
            f"Checkpoint used split_mode={old_mode}; set that explicitly to resume. "
            "To change the split, initialize from a weight file and start a new run."
        )
    for key in ("repo_id", "episodes"):
        if old_lerobot.get(key) != lerobot_config.get(key):
            raise ValueError(
                f"Cannot change lerobot_config.{key} during training resume"
            )
    if old_data.get("train_test_split", 0.95) != train_ratio:
        raise ValueError("Cannot change train_test_split during training resume")
