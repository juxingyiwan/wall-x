"""Training resumes must consume the same batches and random draws."""

import copy
import json
import random
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader, DistributedSampler

from wall_x.robotwin.resume import (
    ReplayableLoader,
    atomic_json,
    capture_rng,
    legacy_cursor,
    read_state,
    restore_rng,
    seeded_cpu,
    validate_legacy_config,
)
from wall_x.robotwin.trainer import RobotwinFSDPTrainer


class RandomDataset:
    def __len__(self):
        return 72

    def __getitem__(self, index):
        return torch.tensor(
            [index, random.random(), np.random.rand(), torch.rand(()).item()]
        )


class RandomCollator:
    def __init__(self):
        self.np_rng = np.random.default_rng()

    def __call__(self, samples):
        return torch.stack(samples), torch.from_numpy(
            self.np_rng.beta(1.5, 1.0, len(samples))
        )


def loader(workers=0, rank=0, epoch=3):
    dataset = RandomDataset()
    sampler = DistributedSampler(
        dataset, num_replicas=2, rank=rank, seed=10222, drop_last=True
    )
    sampler.set_epoch(epoch)
    original = DataLoader(
        dataset,
        batch_size=3,
        sampler=sampler,
        num_workers=workers,
        collate_fn=RandomCollator(),
        persistent_workers=workers > 0,
        prefetch_factor=2 if workers else None,
        drop_last=True,
    )
    return ReplayableLoader(original, 10222, rank, "train")


@pytest.mark.parametrize("workers", [0, 2])
@pytest.mark.parametrize("rank", [0, 1])
def test_resume_matches_batches_noise_and_model_rng_despite_prefetch(workers, rank):
    uninterrupted = loader(workers, rank)
    iterator = iter(uninterrupted)
    for _ in range(5):
        next(iterator)
    assert uninterrupted.next_batch == 5  # Not the prefetched cursor.
    saved_rng = capture_rng()
    expected = [(x, noise, torch.rand(3)) for x, noise in iterator]
    # Building a new loader and arbitrary initialization must not matter.
    torch.manual_seed(42)
    resumed = loader(0 if workers else 2, rank)
    resumed.configure(5)
    restore_rng(saved_rng)
    actual = [(x, noise, torch.rand(3)) for x, noise in resumed]
    assert len(expected) == len(actual) == 7
    for left, right in zip(expected, actual):
        for a, b in zip(left, right):
            assert torch.equal(a, b)


def test_continuous_and_resumed_optimizer_updates_are_identical():
    torch.manual_seed(7)
    model = torch.nn.Linear(4, 1)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 3, gamma=0.9)
    data = loader(2)
    iterator = iter(data)

    def update(batch):
        x, noise = batch
        optimizer.zero_grad()
        loss = ((model(x).flatten() - noise.float() - torch.rand(len(x))) ** 2).mean()
        loss.backward()
        optimizer.step()
        scheduler.step()
        return loss.detach().clone()

    for _ in range(4):
        update(next(iterator))
    states = copy.deepcopy(
        (model.state_dict(), optimizer.state_dict(), scheduler.state_dict())
    )
    rng = capture_rng()
    expected = [update(batch) for batch in iterator]
    final = copy.deepcopy(model.state_dict())
    model.load_state_dict(states[0])
    optimizer.load_state_dict(states[1])
    scheduler.load_state_dict(states[2])
    resumed = loader(0)
    resumed.configure(4)
    restore_rng(rng)
    actual = [update(batch) for batch in resumed]
    assert all(torch.equal(a, b) for a, b in zip(expected, actual))
    assert all(
        torch.equal(value, final[key]) for key, value in model.state_dict().items()
    )


def test_cursor_uses_epoch_shuffle_and_accumulation_boundaries():
    data = loader()
    data.configure(8, accumulation=4)
    assert len(data) == 12 and len(list(data)) == 4
    with pytest.raises(ValueError, match="cursor"):
        data.configure(3, accumulation=2)
    with pytest.raises(ValueError, match="cursor"):
        data.configure(13)
    a = next(iter(loader(epoch=0)))[0][:, 0]
    b = next(iter(loader(epoch=1)))[0][:, 0]
    assert not torch.equal(a, b)


def test_cpu_data_scope_restores_rng_after_exception():
    before = capture_rng()
    with pytest.raises(RuntimeError):
        with seeded_cpu(11):
            torch.rand(10)
            raise RuntimeError("sample failed")
    actual = torch.rand(5)
    restore_rng(before)
    assert torch.equal(actual, torch.rand(5))


@pytest.mark.parametrize(
    "epoch,step,total,expected",
    [
        (0, 2499, 3322, (0, 2500, 2500)),
        (1, 4999, 3322, (1, 1678, 5000)),
        (0, 3321, 3322, (0, 3322, 3322)),
    ],
)
def test_legacy_mid_epoch_cursor(tmp_path, epoch, step, total, expected):
    path = tmp_path / f"{epoch}_{step}"
    path.mkdir()
    for file, value in {
        "global_step": step,
        "current_epoch": epoch,
        "world_size": 8,
    }.items():
        torch.save({file: value}, path / f"{file}.pth")
    assert legacy_cursor(path, range(total), 1, 8) == expected
    with pytest.raises(ValueError, match="GPU count"):
        legacy_cursor(path, range(total), 1, 4)


def test_incomplete_rank_states_fail_closed(tmp_path):
    directory = tmp_path / "robotwin_resume"
    directory.mkdir()
    with pytest.raises(ValueError, match="metadata missing"):
        read_state(tmp_path, 0)
    atomic_json(
        directory / "metadata.json", {"version": 1, "protocol": {"world_size": 2}}
    )
    torch.save(capture_rng(), directory / "rank_00000.pt")
    with pytest.raises(ValueError, match="rank RNG"):
        read_state(tmp_path, 0)
    torch.save(capture_rng(), directory / "rank_00001.pt")
    assert read_state(tmp_path, 1)[0]["version"] == 1


def test_legacy_batch_change_is_rejected(tmp_path):
    config = {
        "task": {},
        "model": {},
        "data": {},
        "hyperparams": {"batch_size_per_gpu": 32},
    }
    (tmp_path / "config.yml").write_text(json.dumps({"_raw_yaml": config}))
    config["hyperparams"]["batch_size_per_gpu"] = 16
    with pytest.raises(ValueError, match="batch_size_per_gpu"):
        validate_legacy_config(tmp_path, SimpleNamespace(_raw_yaml=config))


def test_resume_replays_pending_validation_before_next_batch(monkeypatch):
    trainer = object.__new__(RobotwinFSDPTrainer)
    events = []
    trainer._pending_rng = capture_rng()
    trainer._pending_validation = True
    trainer._pending_epoch_validation = False
    trainer.global_step, trainer.num_training_steps = 5, 6
    trainer.start_epoch, trainer.num_epoch, trainer.nvtx = 0, 1, False
    trainer.cfg = SimpleNamespace(
        checkpoint=SimpleNamespace(resume_from="checkpoint", validate_first=True),
        debug=SimpleNamespace(profile=False),
        logging=SimpleNamespace(epoch_save_interval=10),
    )
    trainer.val_loop = lambda: events.append(("validation", trainer.global_step))

    def train(epoch, profiler):
        events.append("training")
        trainer.global_step = 6

    trainer.train_loop = train
    trainer.save_checkpoint = lambda *a: None
    monkeypatch.setattr("wall_x.robotwin.trainer.barrier", lambda: None)
    trainer.fit()
    assert events == [("validation", 5), "training"]


def test_completed_run_does_not_execute_an_extra_update(monkeypatch):
    trainer = object.__new__(RobotwinFSDPTrainer)
    trainer._pending_rng = None
    trainer._pending_validation = trainer._pending_epoch_validation = False
    trainer.global_step = trainer.num_training_steps = 20
    trainer.cfg = SimpleNamespace(
        checkpoint=SimpleNamespace(resume_from="checkpoint", validate_first=False)
    )
    monkeypatch.setattr("wall_x.robotwin.trainer.barrier", lambda: None)
    trainer.fit()  # Would fail on missing train_loop attributes if entered.


def test_epoch_end_pending_validation_occurs_once(monkeypatch):
    trainer = object.__new__(RobotwinFSDPTrainer)
    trainer._pending_rng = None
    trainer._pending_validation = False
    trainer._pending_epoch_validation = True
    trainer.global_step, trainer.num_training_steps = 12, 13
    trainer.start_epoch, trainer.num_epoch, trainer.nvtx = 1, 2, False
    trainer.cfg = SimpleNamespace(
        checkpoint=SimpleNamespace(resume_from="checkpoint", validate_first=True),
        debug=SimpleNamespace(profile=False),
        logging=SimpleNamespace(epoch_save_interval=10),
    )
    events = []
    trainer.val_loop = lambda: events.append(("validation", trainer.global_step))

    def train(epoch, profiler):
        events.append(("train", epoch))
        trainer.global_step = 13

    trainer.train_loop = train
    trainer.save_checkpoint = lambda *a: None
    monkeypatch.setattr("wall_x.robotwin.trainer.barrier", lambda: None)
    trainer.fit()
    assert events == [("validation", 12), ("train", 1)]


@pytest.mark.parametrize("inside,expected", [(True, 9), (False, 9)])
def test_save_captures_completed_updates_cursor_and_rank_rng(
    tmp_path, monkeypatch, inside, expected
):
    trainer = object.__new__(RobotwinFSDPTrainer)
    trainer.rank, trainer.global_step = 0, 9
    trainer.cfg = SimpleNamespace(checkpoint=SimpleNamespace(save_path=str(tmp_path)))
    trainer._inside_train_loop = inside
    trainer._resume_protocol = {"world_size": 1}
    trainer.train_dataloader = loader()
    trainer.train_dataloader.next_batch = 5
    trainer._should_validate = lambda: True
    monkeypatch.setattr("wall_x.robotwin.trainer.barrier", lambda: None)
    monkeypatch.setattr(
        "wall_x.trainer.fsdp_trainer.fsdp_trainer.FSDPTrainer.save_checkpoint",
        lambda *args: torch.rand(3),
    )
    before = capture_rng()
    trainer.save_checkpoint(0, 9)
    actual_draws = torch.rand(4)
    metadata, rng = read_state(tmp_path / "0_9", 0)
    assert metadata["completed_updates"] == expected
    assert metadata["next_batch"] == 5
    assert metadata["pending_validation"] == inside
    restore_rng(before)
    assert torch.equal(actual_draws, torch.rand(4))
    assert torch.equal(before["torch"], rng["torch"])
