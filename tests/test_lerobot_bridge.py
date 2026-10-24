"""Contract checks for scenario-facing LeRobot loader construction."""

import torch
from torch.utils.data import DataLoader, DistributedSampler

from wall_x.data.backends.lerobot.bridge import wrap_dataset


def test_wrapping_preserves_split_iterators_rng_and_epoch_seeding():
    rows = list(range(24))
    sampler = DistributedSampler(rows, num_replicas=1, rank=0, seed=17)
    train = DataLoader(rows, batch_size=3, sampler=sampler)
    validation = DataLoader([100, 101, 102], batch_size=1)
    rng = torch.get_rng_state().clone()
    dataset = wrap_dataset(
        object(),
        train,
        sampler,
        len(train),
        val_dataloader=validation,
        val_num=len(validation),
    )
    assert torch.equal(rng, torch.get_rng_state())
    first_epoch = list(sampler)
    dataset.set_epoch(3)
    expected = DistributedSampler(rows, num_replicas=1, rank=0, seed=17)
    expected.set_epoch(3)
    assert list(sampler) == list(expected) != first_epoch
    assert dataset.global_train_iters.value == len(train)
    assert dataset.global_val_iters.value == len(validation)

    train_iter = iter(dataset.get_train_dataloader())
    observed = next(train_iter).tolist()
    assert [batch.item() for batch in dataset.get_val_dataloader()] == [100, 101, 102]
    observed.extend(value for batch in train_iter for value in batch.tolist())
    assert observed == list(expected)


def test_wrapping_without_validation_or_sampler():
    train = DataLoader([1, 2, 3], batch_size=1)
    dataset = wrap_dataset(object(), train, None, len(train))
    dataset.set_epoch(7)
    assert dataset.get_val_dataloader() is None
    assert dataset.global_val_iters.value == 0
    assert [batch.item() for batch in dataset.get_train_dataloader()] == [1, 2, 3]
