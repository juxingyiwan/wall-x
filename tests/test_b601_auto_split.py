import copy
import json

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import torch
import yaml
from b601_training_fixture import Harness, fixture_config
from test_b601_pipeline import episodes

from wall_x.config import load_config
from wall_x.data.b601 import digest, read_split, validate_checkpoint
from wall_x.data.b601_split import resolve_split, stratified_split
from wall_x.data.backends.lerobot.build import (
    _build_flat_config,
    load_lerobot_normalizers,
)


def setup_config(root):
    fixture_config(root)
    numeric = root / "data" / "part.parquet"
    numeric.parent.mkdir()
    ids, tasks = [], []
    for ep, values in episodes().items():
        ids.extend([ep] * len(values["task_index"]))
        tasks.extend(values["task_index"].tolist())
    pq.write_table(pa.table({"episode_index": ids, "task_index": tasks}), numeric)
    raw = yaml.safe_load((root / "config.yml").read_text())
    raw["data"]["lerobot_config"]["b601"]["validation_episodes_per_task"] = 1
    expected = stratified_split(episodes(), seed=10222, validation_episodes_per_task=1)
    (root / "split.json").write_text(json.dumps(expected))
    (root / "config.yml").write_text(yaml.safe_dump(raw, sort_keys=False))
    return raw, expected


def test_automatic_split_matches_file_without_writes_or_rng_changes(tmp_path):
    import numpy as np

    raw, expected = setup_config(tmp_path)
    del raw["data"]["lerobot_config"]["b601"]["split_manifest"]
    (tmp_path / "split.json").unlink()
    before = sorted(tmp_path.rglob("*"))
    statistics = (tmp_path / "norm_stats.json").read_bytes()
    torch_state, np_state = torch.get_rng_state(), np.random.get_state()
    result = resolve_split(raw)
    assert result == expected
    assert digest(result) == digest(expected)
    result["train_episodes"].clear()
    assert resolve_split(raw) == expected  # Cached result is not mutable by callers.
    assert sorted(tmp_path.rglob("*")) == before
    assert (tmp_path / "norm_stats.json").read_bytes() == statistics
    assert torch.equal(torch_state, torch.get_rng_state())
    assert np.array_equal(np_state[1], np.random.get_state()[1])
    (tmp_path / "config.yml").write_text(yaml.safe_dump(raw, sort_keys=False))
    assert load_lerobot_normalizers(load_config(str(tmp_path / "config.yml")))
    raw["hyperparams"]["seed"] = 1742
    (tmp_path / "config.yml").write_text(yaml.safe_dump(raw, sort_keys=False))
    flat, _ = _build_flat_config(load_config(str(tmp_path / "config.yml")))
    assert resolve_split(flat) == resolve_split(raw)


def test_parameters_control_split_and_explicit_missing_file_is_not_ignored(tmp_path):
    raw, expected = setup_config(tmp_path)
    (tmp_path / "split.json").unlink()
    with pytest.raises(FileNotFoundError):
        resolve_split(raw)
    opts = raw["data"]["lerobot_config"]["b601"]
    del opts["split_manifest"]
    raw["hyperparams"]["seed"] += 1
    assert digest(resolve_split(raw)) != digest(expected)
    opts["validation_episodes_per_task"] = 2
    with pytest.raises(ValueError, match="retain at least one"):
        resolve_split(raw)
    del opts["validation_episodes_per_task"]
    with pytest.raises(ValueError, match="validation_episodes_per_task"):
        resolve_split(raw)


def test_auto_split_rejects_mixed_task_episode(tmp_path):
    raw, _ = setup_config(tmp_path)
    del raw["data"]["lerobot_config"]["b601"]["split_manifest"]
    resolve_split(raw)
    pq.write_table(
        pa.table({"episode_index": [0, 0], "task_index": [0, 1]}),
        tmp_path / "data/part.parquet",
    )
    with pytest.raises(ValueError, match="one task"):
        resolve_split(raw)


def test_file_to_automatic_resume_and_checkpoint_configuration_are_portable(tmp_path):
    raw, expected = setup_config(tmp_path)
    cfg = load_config(str(tmp_path / "config.yml"))
    continuous = Harness(copy.deepcopy(cfg))
    continuous.cfg.checkpoint.save_path = str(tmp_path / "continuous")
    continuous.fit()
    reference = continuous.parameters_snapshot()
    before = Harness(copy.deepcopy(cfg))
    before.cfg.checkpoint.save_path = str(tmp_path / "interrupted")
    before.num_training_steps = 2
    before.fit()
    checkpoint = tmp_path / "interrupted/0_2"
    del raw["data"]["lerobot_config"]["b601"]["split_manifest"]
    (tmp_path / "split.json").unlink()
    (tmp_path / "config.yml").write_text(yaml.safe_dump(raw, sort_keys=False))
    resumed = Harness(load_config(str(tmp_path / "config.yml")), resume=checkpoint)
    resumed.cfg.checkpoint.save_path = str(tmp_path / "resumed")
    resumed.fit()
    assert before.trace + resumed.trace == continuous.trace
    for a, b in zip(reference, resumed.parameters_snapshot()):
        torch.testing.assert_close(a, b, atol=0, rtol=0)
    assert not (tmp_path / "split.json").exists()
    saved = yaml.safe_load((checkpoint / "config.yml").read_text())
    assert isinstance(saved["data"]["lerobot_config"]["b601"]["split_manifest"], dict)
    saved["data"]["lerobot_config"]["root"] = "/unavailable/dataset"
    assert resolve_split(saved) == expected
    assert read_split(checkpoint / "episode_split.json") == expected
    assert validate_checkpoint(saved, checkpoint)
