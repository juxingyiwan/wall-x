"""Full loading must avoid per-frame transforms without changing action queries."""

import json

import numpy as np
import pytest
import torch
from lerobot.datasets import lerobot_dataset as lerobot_module

from wall_x.robotwin import dataset as loader


@pytest.fixture(autouse=True)
def restore_video_reader_after_each_test(monkeypatch):
    import torchvision

    monkeypatch.setattr(torchvision.io, "VideoReader", torchvision.io.VideoReader)


@pytest.fixture
def tiny_lerobot(tmp_path):
    root = tmp_path / "dataset"
    dataset = lerobot_module.LeRobotDataset.create(
        repo_id="local/loading-test",
        root=root,
        fps=50,
        use_videos=False,
        features={
            "observation.state": {
                "dtype": "float32",
                "shape": (2,),
                "names": ["state_a", "state_b"],
            },
            "action": {
                "dtype": "float32",
                "shape": (2,),
                "names": ["action_a", "action_b"],
            },
        },
    )
    for episode in range(3):
        for frame in range(3):
            value = np.array(
                [episode * 100 + frame, -(episode * 100 + frame)], dtype=np.float32
            )
            dataset.add_frame(
                {"observation.state": value, "action": value, "task": "test"}
            )
        dataset.save_episode()
    dataset.finalize()
    return root


@pytest.mark.parametrize("episodes", [None, [0, 1, 2], [0, 2]])
def test_full_and_filtered_loading_preserve_split_and_action_alignment(
    monkeypatch, tmp_path, tiny_lerobot, episodes
):
    subset = episodes == [0, 2]
    split = {"train": [0] if subset else [0, 2], "validation": [2] if subset else [1]}
    path = tmp_path / "split.json"
    path.write_text(json.dumps(split))
    monkeypatch.setenv("LOCAL_RANK", "0")
    monkeypatch.setenv("LOCAL_WORLD_SIZE", "1")
    monkeypatch.setattr(torch.distributed, "get_rank", lambda: 0)
    monkeypatch.setattr(torch.distributed, "barrier", lambda **kwargs: None)
    monkeypatch.setattr(loader, "load_norm_stats", lambda *args, **kwargs: {})
    calls = []
    original = lerobot_module.hf_transform_to_torch

    def track_transform(batch):
        calls.append(len(batch.get("episode_index", batch.get("index", []))))
        return original(batch)

    monkeypatch.setattr(lerobot_module, "hf_transform_to_torch", track_transform)
    config = {
        "norm_stats_path": "unused",
        "batch_size_per_gpu": 1,
        "data": {
            "episode_split_path": str(path),
            "action_horizon": 3,
            "key_mappings": {
                "camera": {},
                "state": "observation.state",
                "action": "action",
            },
        },
    }
    wrapped, _ = loader.load_robotwin_data(
        config,
        {
            "repo_id": "local/loading-test",
            "root": str(tiny_lerobot),
            "episodes": episodes,
        },
    )
    dataset = wrapped.hf_dataset
    import torchvision

    assert torchvision.io.VideoReader._robotwin_bounded_threads
    if subset:
        assert dataset.episodes == [0, 2]
        assert dataset._absolute_to_relative_idx == {0: 0, 1: 1, 2: 2, 6: 3, 7: 4, 8: 5}
        assert wrapped.train_dataset.indices == [0, 1, 2]
        assert wrapped.val_dataset.indices == [3, 4, 5]
    else:
        assert dataset.episodes is None
        assert dataset._absolute_to_relative_idx is None
        assert calls == []  # No O(frame count) torch transform during initialization.
        assert wrapped.train_dataset.indices == [0, 1, 2, 6, 7, 8]
        assert wrapped.val_dataset.indices == [3, 4, 5]
    assert dataset.hf_dataset.format["type"] == "custom"
    # The final episode must query its own actions even in a noncontiguous subset.
    start = 3 if subset else 6
    torch.testing.assert_close(
        dataset[start]["action"],
        torch.tensor([[200.0, -200.0], [201.0, -201.0], [202.0, -202.0]]),
    )
    last = dataset[start + 2]
    torch.testing.assert_close(
        last["action"],
        torch.tensor([[202.0, -202.0], [202.0, -202.0], [202.0, -202.0]]),
    )
    assert last["action_is_pad"].tolist() == [False, True, True]


@pytest.mark.parametrize("adapter", [None, "robotwin"])
def test_bridge_only_selects_robotwin_when_explicit(monkeypatch, adapter):
    import importlib
    from types import SimpleNamespace

    bridge = importlib.import_module(
        "wall_x.robotwin.build"
        if adapter == "robotwin"
        else "wall_x.data.backends.lerobot.build"
    )
    shared = importlib.import_module("wall_x.data.backends.lerobot.loader")
    selected = []

    class SelectedLoader(Exception):
        pass

    def stop(name):
        def load(*args, **kwargs):
            selected.append(name)
            raise SelectedLoader

        return load

    monkeypatch.setattr(
        bridge, "_build_flat_config", lambda cfg: ({}, {"adapter": adapter})
    )
    monkeypatch.setattr(shared, "load_lerobot_data", stop("lerobot"))
    if adapter == "robotwin":
        monkeypatch.setattr(bridge, "load_robotwin_data", stop("robotwin"))
    with pytest.raises(SelectedLoader):
        entry = (
            bridge.load_robotwin_v2 if adapter == "robotwin" else bridge.load_lerobot_v2
        )
        entry(
            SimpleNamespace(
                hyperparams=SimpleNamespace(seed=42),
                checkpoint=SimpleNamespace(resume_from=None),
                data=SimpleNamespace(train_test_split=0.95),
            )
        )
    assert selected == [adapter or "lerobot"]
