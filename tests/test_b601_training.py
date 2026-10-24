import copy
import json

import pytest
import torch
import yaml
from b601_training_fixture import Harness, fixture_config

from wall_x.config import load_config
from wall_x.data.b601 import LAYOUT
from wall_x.data.backends.lerobot.build import load_lerobot_normalizers
from wall_x.trainer.adapters.b601_state import canonicalize_config, load_runtime


@pytest.mark.parametrize("accumulation", [1, 2])
def test_real_loop_save_resume_matches_uninterrupted(tmp_path, accumulation):
    cfg = fixture_config(tmp_path, accumulation)
    continuous = Harness(copy.deepcopy(cfg))
    continuous.cfg.checkpoint.save_path = str(tmp_path / "continuous")
    continuous.fit()
    reference = continuous.parameters_snapshot()
    before = Harness(copy.deepcopy(cfg))
    before.cfg.checkpoint.save_path = str(tmp_path / "interrupted")
    before.num_training_steps = 2
    before.fit()
    checkpoint = tmp_path / "interrupted/0_2"
    state = load_runtime(checkpoint, 0, 1)
    assert state["completed_updates"] == 2
    assert state["next_epoch"] == 0 and state["next_batch"] == 2 * accumulation
    assert (
        torch.load(checkpoint / "global_step.pth", weights_only=True)["global_step"]
        == 2
    )
    resumed = Harness(copy.deepcopy(cfg), resume=checkpoint)
    resumed.cfg.checkpoint.save_path = str(tmp_path / "resumed")
    resumed.fit()
    assert resumed.global_step == 8
    assert continuous.trace == before.trace + resumed.trace
    for left, right in zip(reference, resumed.parameters_snapshot()):
        torch.testing.assert_close(left, right, atol=0, rtol=0)
    assert continuous.lr_scheduler.state_dict() == resumed.lr_scheduler.state_dict()
    # Real native metadata save -> typed training reload -> actual normalizers.
    loaded = load_config(str(checkpoint / "config.yml"))
    assert list(loaded.task.dof_config.items()) == list(LAYOUT.items())
    assert load_lerobot_normalizers(loaded) is not None


@pytest.mark.parametrize("failure", ["nan", "inf", "gradient"])
def test_nonfinite_stops_before_any_optimizer_update(tmp_path, failure):
    t = Harness(fixture_config(tmp_path), failure=failure)
    weights = t.parameters_snapshot()
    with pytest.raises(FloatingPointError, match="Non-finite"):
        t.fit()
    assert t.global_step == 0
    assert not t.optimizer.state
    for left, right in zip(weights, t.parameters_snapshot()):
        torch.testing.assert_close(left, right, atol=0, rtol=0)


def test_sorted_legacy_yaml_is_canonicalized_only_for_known_layout(tmp_path):
    cfg = fixture_config(tmp_path)
    path = tmp_path / "config.yml"
    path.write_text(yaml.safe_dump(yaml.safe_load(path.read_text()), sort_keys=True))
    cfg = load_config(str(path))
    canonicalize_config(cfg)
    assert load_lerobot_normalizers(cfg) is not None
    cfg.task.dof_config["action_padding"] = 18
    with pytest.raises(ValueError, match="known"):
        canonicalize_config(cfg)


def test_resume_rejects_changed_seed_and_incomplete_checkpoint(tmp_path):
    cfg = fixture_config(tmp_path)
    before = Harness(copy.deepcopy(cfg))
    before.num_training_steps = 2
    before.fit()
    checkpoint = tmp_path / "ckpts/0_2"
    changed = copy.deepcopy(cfg)
    changed.hyperparams.seed += 1
    with pytest.raises(ValueError, match="contract mismatch"):
        Harness(changed, resume=checkpoint)
    marker = checkpoint / "b601_training_state.json"
    meta = json.loads(marker.read_text())
    meta["completed_updates"] += 1
    marker.write_text(json.dumps(meta))
    with pytest.raises(ValueError, match="progress mismatch"):
        load_runtime(checkpoint, 0, 1)
    marker.unlink()
    with pytest.raises(ValueError, match="lacks exact"):
        load_runtime(checkpoint, 0, 1)


@pytest.mark.parametrize("stop", [4, 6])
def test_periodic_and_epoch_boundary_checkpoint_counts(tmp_path, stop):
    cfg = fixture_config(tmp_path)
    cfg.logging.save_interval = 2
    t = Harness(copy.deepcopy(cfg))
    t.num_training_steps = stop
    t.fit()
    checkpoint = tmp_path / f"ckpts/0_{stop}"
    state = load_runtime(checkpoint, 0, 1)
    assert state["completed_updates"] == stop
    assert state["next_epoch"] == stop // 6
    assert state["next_batch"] == stop % 6
    for path in (tmp_path / "ckpts").iterdir():
        count = int(path.name.split("_")[1])
        assert load_runtime(path, 0, 1)["completed_updates"] == count
    cfg.logging.save_interval = 0
    resumed = Harness(cfg, resume=checkpoint)
    resumed.cfg.checkpoint.save_path = str(tmp_path / "resumed")
    assert resumed.global_step == stop
    resumed.fit()
    assert resumed.global_step == 8


def test_accumulation_tail_is_not_loaded_or_applied(tmp_path):
    t = Harness(fixture_config(tmp_path, accumulation=4, updates=1))
    t.fit()
    assert len(t.trace) == 4  # Six loader batches, one full four-batch update.
    assert t.global_step == 1 and t.micro_step == 0
    state = load_runtime(tmp_path / "ckpts/0_1", 0, 1)
    assert (state["next_epoch"], state["next_batch"]) == (1, 0)


def test_validation_does_not_change_training_rng_or_mode(tmp_path, monkeypatch):
    import random

    import numpy as np

    from wall_x.trainer.fsdp_trainer.fsdp_trainer import FSDPTrainer

    cfg = fixture_config(tmp_path, updates=3)
    continuous = Harness(copy.deepcopy(cfg))
    continuous.fit()

    def stochastic_validation(self):
        self.model.eval()
        random.random()
        np.random.rand(23)
        torch.rand(31)

    monkeypatch.setattr(FSDPTrainer, "val_loop", stochastic_validation)
    cfg.logging.val_interval = 1
    cfg.checkpoint.save_path = str(tmp_path / "with-validation")
    validated = Harness(cfg)
    validated.fit()
    assert validated.model.training
    assert validated.trace == continuous.trace
    for a, b in zip(validated.parameters_snapshot(), continuous.parameters_snapshot()):
        torch.testing.assert_close(a, b, atol=0, rtol=0)
