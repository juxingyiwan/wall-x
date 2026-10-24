import json

import numpy as np
import pytest

from wall_x.robotwin.episode_split import (
    episode_slices,
    load_episode_split,
    split_frame_indices,
)


def test_split_keeps_every_frame_of_an_episode_together(tmp_path):
    path = tmp_path / "split.json"
    path.write_text(json.dumps({"train": [2, 0], "validation": [1]}))
    train, validation = split_frame_indices([0, 0, 1, 1, 1, 2], path)
    assert train == [0, 1, 5]
    assert validation == [2, 3, 4]


@pytest.mark.parametrize(
    "data",
    [
        {"train": [0], "validation": [0]},
        {"train": [], "validation": [0]},
        {"train": [0, 0], "validation": [1]},
        {"train": [True], "validation": [1]},
        {"train": [0], "validation": [2]},
    ],
)
def test_invalid_splits_are_rejected(tmp_path, data):
    path = tmp_path / "split.json"
    path.write_text(json.dumps(data))
    with pytest.raises(ValueError):
        load_episode_split(path, [0, 1])


def test_episode_order_is_validated():
    assert episode_slices([2, 2, 4], [0, 1, 0]) == [slice(0, 2), slice(2, 3)]
    with pytest.raises(ValueError):
        episode_slices([2, 4, 2])
    with pytest.raises(ValueError):
        episode_slices([2, 2, 4], [1, 0, 0])


def test_loader_subsets_follow_manifest(tmp_path):
    from wall_x.robotwin.dataset import (
        RobotwinPreprocessedDataset as PreprocessedDataset,
    )

    class FakeDataset:
        hf_dataset = {"episode_index": [0, 0, 1, 1, 1, 2]}

        def __len__(self):
            return 6

    path = tmp_path / "split.json"
    path.write_text(json.dumps({"train": [0, 2], "validation": [1]}))
    config = {
        "episode_split_path": str(path),
        "train_test_split": 0.95,
        "seed": 42,
        "key_mappings": {
            "camera": {},
            "state": "observation.state",
            "action": "action",
        },
    }
    dataset = PreprocessedDataset(FakeDataset(), {}, {}, config, {})
    assert dataset.train_dataset.indices == [0, 1, 5]
    assert dataset.val_dataset.indices == [2, 3, 4]
    assert len(dataset) == 3
    dataset._eval()
    assert dataset._dataset.indices == [2, 3, 4]
    dataset._train()
    assert dataset._dataset.indices == [0, 1, 5]


def test_relative_statistics_do_not_cross_episode_boundaries():
    from wall_x.robotwin import norm_stats as stats

    state = np.array([[0], [1], [1000], [1002]], dtype=np.float32)
    action = state + 1
    vectors = stats.collect_dof_vectors_from_arrays(
        state,
        action,
        {"pos": [0, 1]},
        {"pos_relative": [0, 1]},
        action_chunk=3,
        episode_ids=np.array([0, 0, 1, 1]),
        frame_indices=np.array([0, 1, 0, 1]),
    )
    values = vectors["action"]["pos_relative"].ravel()
    # Two episodes x two anchors x three actions, with terminal-action padding.
    assert len(values) == 12
    assert values.max() == 3
    assert values.min() == 1


def test_norm_stats_exclude_validation_outliers(monkeypatch, tmp_path):
    from wall_x.robotwin import norm_stats as stats

    path = tmp_path / "split.json"
    path.write_text(json.dumps({"train": [0], "validation": [1]}))
    table = {
        "observation.state": [[1], [2], [10000]],
        "action": [[3], [4], [-10000]],
        "episode_index": [0, 0, 1],
        "frame_index": [0, 1, 0],
    }
    monkeypatch.setattr(
        stats,
        "_load_state_action_table",
        lambda *args: (table, "observation.state", "action"),
    )
    result = stats.compute_norm_stats_with_dof_config(
        tmp_path,
        tmp_path / "stats.json",
        {"pos": [0, 1]},
        {"pos": [0, 1]},
        episode_split_path=str(path),
    )
    np.testing.assert_allclose(result["observation.state"]["mean"], [1.5])
    np.testing.assert_allclose(result["action"]["mean"], [3.5])
