import json

import pytest
from torch.utils.data import default_collate

from wall_x.data.backends.lerobot import loader
from wall_x.data.backends.lerobot.episode_split import (
    episode_split,
    validate_resume_split,
    validate_stats_split,
)


class ScalarDataset(loader.PreprocessedDataset):
    def __getitem__(self, index):
        return self._dataset[index]


@pytest.mark.parametrize("workers", [0, 1])
def test_interleaved_validation_preserves_training_iterator(monkeypatch, workers):
    monkeypatch.setattr(loader, "DataCollator", lambda *args: default_collate)
    data = ScalarDataset(
        list(range(12)),
        {"num_workers": workers, "batch_size_per_gpu": 1},
        {},
        {"train_test_split": 0.8, "seed": 42, "key_mappings": {"camera": {}}},
        {"repo_id": "test"},
        val_dataset=list(range(100, 103)),
    )
    train, sampler = data.get_train_dataloader()
    iterator = iter(train)
    first = next(iterator).item()
    val, _ = data.get_val_dataloader()
    assert [x.item() for x in val] == [100, 101, 102]
    assert sorted([first] + [x.item() for x in iterator]) == list(range(12))
    sampler.set_epoch(1)
    assert sorted(x.item() for x in train) == list(range(12))
    assert train.dataset is not val.dataset


@pytest.mark.parametrize("ratio", [0, -1, 1.1, float("nan")])
def test_invalid_ratios_fail(ratio):
    with pytest.raises(ValueError):
        episode_split(10, {"repo_id": "test"}, ratio)


def test_episode_order_subset_and_ratio():
    split = episode_split(10, {"repo_id": "test", "episodes": [9, 2, 7, 0, 4]}, 0.8)
    assert split["train_episodes"] == [9, 2, 7, 0]
    assert split["val_episodes"] == [4]
    assert episode_split(1, {"repo_id": "test"}, 1)["val_episodes"] == []


def test_actual_loader_loads_held_out_episodes(monkeypatch, tmp_path):
    from types import SimpleNamespace

    calls = []

    class FakeLeRobot:
        def __init__(self, **kwargs):
            calls.append(kwargs["episodes"])
            self.episodes = kwargs["episodes"]
            self.num_episodes = len(self.episodes)
            self.num_frames = len(self.episodes) * 10

        def __len__(self):
            return self.num_frames

    config = {"repo_id": "test"}
    manifest = episode_split(10, config, 0.8)
    stats = tmp_path / "stats.json"
    stats.write_text(json.dumps({"episode_split": manifest}))
    monkeypatch.setattr(
        loader,
        "LeRobotDatasetMetadata",
        lambda *a, **kw: SimpleNamespace(fps=20, total_episodes=10),
    )
    monkeypatch.setattr(loader, "LeRobotDataset", FakeLeRobot)
    monkeypatch.setattr(loader, "load_norm_stats", lambda *a, **kw: {})
    dataset, _ = loader.load_lerobot_data(
        {
            "norm_stats_path": str(stats),
            "batch_size_per_gpu": 1,
            "data": {
                "train_test_split": 0.8,
                "seed": 42,
                "key_mappings": {"camera": {}, "action": "action"},
            },
        },
        config,
    )
    assert calls == [list(range(8)), [8, 9]]
    assert len(dataset.train_dataset) == 80
    assert len(dataset.val_dataset) == 20


def test_stats_and_resume_cannot_silently_change_split(tmp_path):
    import yaml

    config = {"repo_id": "test"}
    manifest = episode_split(10, config, 0.8)
    path = tmp_path / "stats.json"
    path.write_text(json.dumps({"episode_split": manifest}))
    validate_stats_split(path, manifest)
    with pytest.raises(ValueError, match="Recompute"):
        validate_stats_split(path, episode_split(10, config, 0.9))
    (tmp_path / "config.yml").write_text(
        yaml.safe_dump({"data": {"lerobot_config": config, "train_test_split": 0.8}})
    )
    with pytest.raises(ValueError, match="legacy_frame"):
        validate_resume_split(tmp_path, config, 0.8)
    validate_resume_split(tmp_path, dict(config, split_mode="legacy_frame"), 0.8)
