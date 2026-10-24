import copy
import json

import pytest
import torch
import yaml
from b601_training_fixture import Harness
from test_b601_auto_split import setup_config
from test_b601_pipeline import episodes

from wall_x.config import load_config
from wall_x.data.b601 import digest, read_split, validate_checkpoint
from wall_x.data.b601_prepare import prepare, prepare_split
from wall_x.data.b601_split import resolve_split, stratified_split


def groups(sizes):
    result = {}
    for task, size in enumerate(sizes):
        for _ in range(size):
            result[len(result)] = {"task_index": [task]}
    return result


@pytest.mark.parametrize(
    "ratio,expected",
    [
        (0, [0, 0, 0]),
        (1e-7, [0, 0, 0]),
        (1e-6, [0, 0, 0]),
        (1.000001e-6, [1, 1, 1]),
        (0.01, [1, 1, 1]),
        (0.05, [2, 5, 2]),
        (0.29, [14, 29, 14]),
        (0.58, [29, 58, 29]),
        (0.9999999, [49, 99, 49]),
    ],
)
def test_ratio_counts_boundaries_and_disjoint_coverage(ratio, expected):
    ep = groups([50, 100, 50])
    split = stratified_split(ep, validation_ratio=ratio)
    assert split == stratified_split(ep, validation_ratio=ratio)
    counts = [
        sum(ep[e]["task_index"][0] == task for e in split["validation_episodes"])
        for task in range(3)
    ]
    assert counts == expected
    assert sorted(split["train_episodes"] + split["validation_episodes"]) == list(ep)
    assert not set(split["train_episodes"]) & set(split["validation_episodes"])
    assert read_split(split) == split


@pytest.mark.parametrize(
    "ratio", [-0.01, 1.0, 1.1, float("nan"), float("inf"), True, "0.01"]
)
def test_invalid_ratio_is_rejected(ratio):
    with pytest.raises(ValueError, match="validation_ratio"):
        stratified_split(groups([50]), validation_ratio=ratio)


def test_single_episode_and_conflicting_parameters():
    assert stratified_split(groups([1]), validation_ratio=0)["train_episodes"] == [0]
    with pytest.raises(ValueError, match="retain at least one training"):
        stratified_split(groups([1]), validation_ratio=0.01)
    with pytest.raises(ValueError, match="Set only"):
        stratified_split(
            groups([50]), validation_ratio=0.01, validation_episodes_per_task=1
        )


def set_ratio(raw, ratio):
    opts = raw["data"]["lerobot_config"]["b601"]
    opts.pop("split_manifest", None)
    opts.pop("validation_episodes_per_task", None)
    opts["validation_ratio"] = ratio


def test_ratio_one_percent_preserves_split_checksum_and_exact_resume(tmp_path):
    raw, old_split = setup_config(tmp_path)
    cfg = load_config(str(tmp_path / "config.yml"))
    reference = Harness(copy.deepcopy(cfg))
    reference.cfg.checkpoint.save_path = str(tmp_path / "reference")
    reference.fit()
    before = Harness(copy.deepcopy(cfg))
    before.num_training_steps = 2
    before.fit()
    set_ratio(raw, 0.01)
    assert resolve_split(raw) == old_split
    assert digest(resolve_split(raw)) == digest(old_split)
    (tmp_path / "split.json").unlink()
    (tmp_path / "config.yml").write_text(yaml.safe_dump(raw, sort_keys=False))
    resumed = Harness(
        load_config(str(tmp_path / "config.yml")), resume=tmp_path / "ckpts/0_2"
    )
    resumed.cfg.checkpoint.save_path = str(tmp_path / "resumed")
    resumed.fit()
    assert reference.trace == before.trace + resumed.trace
    for a, b in zip(reference.parameters_snapshot(), resumed.parameters_snapshot()):
        torch.testing.assert_close(a, b, atol=0, rtol=0)


def test_zero_ratio_checkpoint_and_no_validation_training(tmp_path):
    raw, _ = setup_config(tmp_path)
    set_ratio(raw, 0)
    raw["hyperparams"]["max_optimizer_steps"] = 2
    raw["logging"]["val_interval"] = 1
    (tmp_path / "config.yml").write_text(yaml.safe_dump(raw, sort_keys=False))
    trainer = Harness(load_config(str(tmp_path / "config.yml")))
    trainer.fit()
    assert trainer.global_step == 2
    checkpoint = tmp_path / "ckpts/0_2"
    assert read_split(checkpoint / "episode_split.json")["validation_episodes"] == []
    saved = yaml.safe_load((checkpoint / "config.yml").read_text())
    assert validate_checkpoint(saved, checkpoint)
    assert resolve_split(saved)["train_episodes"] == list(range(6))


def test_preparation_and_training_share_ratio_semantics(tmp_path, monkeypatch):
    raw, _ = setup_config(tmp_path)
    set_ratio(raw, 0)
    monkeypatch.setattr(
        "wall_x.data.b601_prepare.load_episodes", lambda root: (episodes(), [])
    )
    report = prepare_split(raw, tmp_path / "exported.json")
    assert report["validation_episodes"] == 0
    assert read_split(tmp_path / "exported.json") == resolve_split(raw)
    (tmp_path / "meta").mkdir()
    (tmp_path / "meta/info.json").write_text(json.dumps({"fps": 30}))
    output = tmp_path / "ab"
    prepare(raw, output)
    for name in ("absolute", "delta"):
        prepared = yaml.safe_load((output / f"{name}.yml").read_text())
        opts = prepared["data"]["lerobot_config"]["b601"]
        assert opts["validation_ratio"] == 0
        assert "validation_episodes_per_task" not in opts
        assert resolve_split(prepared)["validation_episodes"] == []
